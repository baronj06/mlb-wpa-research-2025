"""Phase 3 guards and storage tests use labeled synthetic fixtures only."""

import json

import pytest
import requests

from mlb_wpa.collection import CachedClient, read_cached_body
from mlb_wpa.final_validation import parse_query
from mlb_wpa.phase3_quality import evaluate_quality, require_metrics_permission
from mlb_wpa.season import canonical_manifest, fallback_rows, RateLimiter


def test_compressed_cache_preserves_original_hash_and_reuses_offline(tmp_path):
    body=b'{"synthetic":true}'
    response=requests.Response();response.status_code=200;response._content=body
    response.url='https://statsapi.mlb.com/api/v1/synthetic'
    class Session:
        calls=0
        def get(self,*args,**kwargs):self.calls+=1;return response
    session=Session();client=CachedClient(tmp_path,session=session,compress=True,sleep=lambda _:None)
    raw,receipt=client.get(response.url)
    assert raw==body and receipt['storage_encoding']=='gzip'
    assert read_cached_body(tmp_path,receipt)==body
    assert CachedClient(tmp_path,offline=True).get(response.url)[0]==body
    assert session.calls==1
    (tmp_path/receipt['storage_path']).write_bytes(b'corrupt gzip')
    with pytest.raises((ValueError,OSError)):
        read_cached_body(tmp_path,receipt)


def test_direct_fangraphs_json_adapter_keeps_observed_values():
    box={'home':{'bat':[]},'away':{'bat':[]},'teamInfo':{}}
    assert parse_query(json.dumps(box).encode(),'scores/box-score')==box
    events=[{'gameSeq':'1','wpa':'.01','li':'1.0'}]
    assert parse_query(json.dumps(events).encode(),'scores/play-log')==events
    with pytest.raises(ValueError):parse_query(b'{"error":"synthetic"}','scores/box-score')


def test_canonical_schedule_prefers_final_postponed_occurrence():
    old={'gamePk':1,'gameType':'R','officialDate':'2025-04-05','doubleHeader':'N','gameNumber':1,
         'status':{'abstractGameState':'Preview'},
         'teams':{'home':{'team':{'id':111,'name':'Red Sox'}},'away':{'team':{'name':'Cardinals'}}}}
    final={**old,'officialDate':'2025-04-06','doubleHeader':'S','status':{'abstractGameState':'Final'}}
    result=canonical_manifest({'dates':[{'games':[old]},{'games':[final]}]})
    assert len(result['games'])==1 and result['games'][0]['date']=='2025-04-06'
    assert result['games'][0]['dh']==1


def test_reference_circuit_prevents_requests_during_cooldown(tmp_path):
    limiter=RateLimiter(interval=0,reference_interval=0)
    limiter.block_reference(300)
    class Session:
        calls=0
        def get(self,*args,**kwargs):self.calls+=1;raise AssertionError('No HTTP allowed')
    session=Session();client=CachedClient(tmp_path,session=session,limiter=limiter,sleep=lambda _:None)
    with pytest.raises(RuntimeError,match='cooldown'):
        client.get('https://www.fangraphs.com/api/synthetic')
    assert session.calls==0 and client.network_requests==0


def test_fallback_never_promotes_alternative_wpa_or_runner_only_pa():
    batter={'about':{'atBatIndex':0},'credits':[{'credit':'b_pa','player':{'id':1}}],
            'result':{'eventType':'walk'},'homeTeamWinProbabilityAdded':5.0}
    runner={'about':{'atBatIndex':1},'credits':[],'result':{'eventType':'caught_stealing_2b'}}
    rows=fallback_rows({'game_pk':1,'date':'2025-04-01'},{'liveData':{'plays':{'allPlays':[batter,runner]}}},'reference_rate_limited')
    assert len(rows)==1 and rows[0]['batter_attributed_wpa'] is None and rows[0]['pre_pa_li'] is None
    assert rows[0]['validated'] is False


def test_quality_gate_accepts_complete_low_loss_snapshot():
    report=evaluate_quality(2,2,2,1000,995,1000)
    assert report['metrics_and_rankings_allowed']
    require_metrics_permission(report)


@pytest.mark.parametrize('change,reason',[
    ({'reference_games':1},'reference_collection_incomplete'),
    ({'official_games':1},'official_game_collection_incomplete'),
    ({'represented_pa':999},'official_pa_representation_mismatch'),
    ({'aggregate_mismatches':1},'official_player_totals_disagree'),
    ({'identity_mismatches':1},'player_pa_identity_counts_disagree'),
    ({'validated_pa':970},'missing_rate_materially_worse_than_pilot')])
def test_quality_failure_stops_rankings(change,reason):
    values=dict(official_games=2,reference_games=2,expected_games=2,official_pa=1000,validated_pa=995,represented_pa=1000)
    values.update(change)
    report=evaluate_quality(**values)
    assert reason in report['stop_reasons']
    with pytest.raises(RuntimeError,match='stopped'):
        require_metrics_permission(report)


def test_primary_table_checks_null_policy_and_actual_actor():
    import pandas as pd
    from mlb_wpa.phase3_finalize import validate_primary_table
    data=pd.DataFrame([
        dict(game_pk=1,at_bat_index=0,batter_id=42,validated=True,batter_attributed_wpa=.02,pre_pa_li=1.2),
        dict(game_pk=1,at_bat_index=1,batter_id=43,validated=False,batter_attributed_wpa=None,pre_pa_li=None)])
    validate_primary_table(data)
    data.loc[1,'pre_pa_li']=0
    with pytest.raises(ValueError,match='Excluded PA'):validate_primary_table(data)
    data.loc[1,'pre_pa_li']=None
    data.loc[0,'pre_pa_li']=float('inf')
    with pytest.raises(ValueError,match='Invalid primary'):validate_primary_table(data)


def test_real_official_pa_credits_come_from_wp_response():
    from mlb_wpa.sample import ROOT
    checkpoint=ROOT/'data/checkpoints/season_2025/games/778563.json'
    if not checkpoint.exists():pytest.skip('Season source cache not present')
    record=json.loads(checkpoint.read_text());client=CachedClient(offline=True)
    metadata=json.loads(read_cached_body(client.root,record['sources']['metadata']))
    wp=json.loads(read_cached_body(client.root,record['sources']['wp']))
    assert not any(p.get('credits') for p in metadata['liveData']['plays']['allPlays'])
    rows=fallback_rows({'game_pk':778563,'date':'2025-03-18'},metadata,'synthetic_reference_unavailable',wp)
    official=sum(p.get('stats',{}).get('batting',{}).get('plateAppearances',0)
                 for side in ['home','away'] for p in metadata['liveData']['boxscore']['teams'][side]['players'].values())
    assert len(rows)==official==74
    assert all(r['batter_id'] is not None and r['batter_attributed_wpa'] is None for r in rows)
