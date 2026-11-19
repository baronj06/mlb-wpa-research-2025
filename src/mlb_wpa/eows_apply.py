"""Apply frozen independently estimated Markov WPA/LI to cached 2025 games."""

from collections import Counter, defaultdict
import hashlib
import json

import numpy as np
import pandas as pd

from .collection import CachedClient, read_cached_body, atomic_json, utc_now
from .independent_we import extract_game, attributed_wpa
from .li_candidates import MarkovWE, PotentialLI, load_baseline
from .sample import ROOT

OUT=ROOT/'reports/eows_2025_exploratory'
DATA=ROOT/'data/processed/season_2025_independent_markov_v03.parquet'


class FrozenLI(PotentialLI):
    """Load saved probabilities/normalization without fitting on application data."""
    def __init__(self,model,parents,parameters,normalizer,hierarchical=False):
        self.model=model;self.parents=parents;self.hierarchical=hierarchical
        self.prior_pa=parameters['prior_pa'];self.cache={};self.child=defaultdict(Counter)
        for x in parameters['context_counts']:
            self.child[tuple(x['key'])]=Counter({tuple(y['outcome']):y['count'] for y in x['outcomes']})
        self.normalizer=normalizer


def frozen_models():
    baseline=json.loads((ROOT/'reports/expanded_we_once/model.json').read_text())
    old,parents=load_baseline(baseline)
    parameters=json.loads((ROOT/'reports/li_accuracy_study/model_parameters.json').read_text())
    results=json.loads((ROOT/'reports/li_accuracy_study/results.json').read_text())
    model=MarkovWE(parents,old.training_games)
    model.calibration=np.array(parameters['markov_calibration'])
    model.calibration_games=old.calibration_games
    for key,pmf in model.run_pmf.items():
        if not np.allclose(pmf,parameters['run_pmf'][f'{key[0]}:{key[1]}'],atol=1e-12,rtol=0):
            raise ValueError('Frozen model run distribution changed')
    normalizers=results['li_normalizers_training_only']
    return (model,FrozenLI(model,parents,parameters,normalizers['markov_global']),
            FrozenLI(model,parents,parameters,normalizers['selected_context'],True))


def apply():
    OUT.mkdir(exist_ok=True);cache=OUT/'game_cache';cache.mkdir(exist_ok=True)
    model,global_li,context_li=frozen_models();client=CachedClient(offline=True)
    manifest=json.loads((ROOT/'config/season_2025_games.json').read_text())['games']
    study=json.loads((ROOT/'reports/li_accuracy_study/design.json').read_text())
    roles={pk:role for role,key in [('training','training_games'),('calibration','calibration_games'),
        ('selection_validation','selection_validation_games'),('reference_test','reference_test_games')] for pk in study[key]}
    reference=pd.read_parquet(ROOT/'reports/reference_review_v04/reference_pas.parquet')
    quarantined={(int(r.game_pk),int(r.at_bat_index)):r.reason for r in reference.loc[~reference.validated].itertuples()}
    assert len(quarantined)==35
    dependencies=[ROOT/f'src/mlb_wpa/{n}.py' for n in ['eows_apply','independent_we','li_candidates','reference_review','attribution']]
    dependencies += [ROOT/'reports/li_accuracy_study/model_parameters.json',ROOT/'reports/li_accuracy_study/results.json',
                     ROOT/'reports/expanded_we_once/model.json',ROOT/'reports/reference_review_v04/reference_pas.parquet']
    method_hash=hashlib.sha256(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in dependencies},sort_keys=True).encode()).hexdigest()
    frames=[];receipts={};hits=0
    for i,game in enumerate(manifest):
        pk=game['game_pk'];record=json.loads((ROOT/f'data/checkpoints/season_2025/games/{pk}.json').read_text());src=record['sources']
        receipts[pk]={k:src[k] for k in ['metadata','wp']}
        fingerprint=hashlib.sha256((method_hash+src['metadata']['sha256']+src['wp']['sha256']).encode()).hexdigest()
        parquet=cache/f'{pk}.parquet';receipt=cache/f'{pk}.json'
        if receipt.exists() and parquet.exists() and json.loads(receipt.read_text()).get('fingerprint')==fingerprint:
            saved=json.loads(receipt.read_text())
            if hashlib.sha256(parquet.read_bytes()).hexdigest()!=saved['sha256']:raise ValueError('Derived cache hash mismatch')
            frame=pd.read_parquet(parquet);hits+=1
        else:
            meta=json.loads(read_cached_body(client.root,src['metadata']));plays=json.loads(read_cached_body(client.root,src['wp']))
            official=meta['liveData']['plays']['allPlays']
            if (len(official)!=len(plays) or meta['gameData']['game']['pk']!=pk or
                any(a['about']['atBatIndex']!=b['about']['atBatIndex'] or a['matchup']['batter']['id']!=b['matchup']['batter']['id']
                    or a['result']['eventType']!=b['result']['eventType'] for a,b in zip(official,plays))):
                raise ValueError('Season feed identity failure')
            rows=extract_game(pk,meta,plays,roles.get(pk,'other_application'))
            observations=[]
            for r in rows:
                s=r['start_state'];wpa=li=alternative=None;reasons=[]
                if s is not None:
                    li=global_li.predict(s)[0];alternative=context_li.predict(s)[0]
                if r['supported']:wpa=attributed_wpa(model,r['batter_before'],r['after_state'])
                else:reasons.append(r['reason'] or 'unsupported_attribution')
                if (pk,r['at_bat_index']) in quarantined:
                    reasons.append('reference_unresolved:'+quarantined[(pk,r['at_bat_index'])])
                if wpa is None or li is None:reasons.append('missing_primary_wpa_li')
                elif not np.isfinite(wpa) or not np.isfinite(li) or abs(wpa)>1 or li<0:reasons.append('invalid_primary_wpa_li')
                valid=not reasons
                observations.append({'game_pk':pk,'date':game['date'],'season':2025,'at_bat_index':r['at_bat_index'],
                    'batter_id':r['batter_id'],'result':r['result'],'model_role':r['role'],
                    'metric_eligible':valid,'exclusion_reason':';'.join(reasons),
                    'estimated_batter_wpa':wpa if valid else None,'estimated_pre_pa_li':li if valid else None,
                    'alternative_pre_pa_li':alternative if valid else None,'diagnostic_start_li':li,
                    'inning':s.inning if s is not None else None,'half':s.half if s is not None else None,
                    'outs_before':s.outs if s is not None else None,'bases_before':s.mask if s is not None else None,
                    'model_source':'Frozen independent Markov v0.3; not FanGraphs',
                    'known_reference_unresolved':(pk,r['at_bat_index']) in quarantined})
            frame=pd.DataFrame(observations)
            for column in ['game_pk','season','at_bat_index','batter_id','inning','outs_before']:frame[column]=frame[column].astype('Int64')
            temporary=parquet.with_suffix('.tmp');frame.to_parquet(temporary,index=False);temporary.replace(parquet)
            atomic_json(receipt,{'fingerprint':fingerprint,'sha256':hashlib.sha256(parquet.read_bytes()).hexdigest(),'pa':len(frame)})
        frames.append(frame)
        if (i+1)%200==0:print(f'Frozen application {i+1}/{len(manifest)} games; derived-cache hits {hits}',flush=True)
    data=pd.concat(frames,ignore_index=True)
    if len(data)!=182926 or data.duplicated(['game_pk','at_bat_index']).any():raise ValueError('Season PA representation failed')
    stats=pd.read_csv(ROOT/'data/processed/season_2025_official_player_stats.csv').set_index('player_id')
    counts=data.groupby('batter_id').size()
    if not counts.reindex(stats.index,fill_value=0).eq(stats.PA).all() or not counts.index.isin(stats.index).all():
        raise ValueError('Player PA ownership mismatch')
    excluded=data.loc[~data.metric_eligible]
    if excluded[['estimated_batter_wpa','estimated_pre_pa_li']].notna().any().any():raise ValueError('Excluded primary values not null')
    old=pd.read_csv(ROOT/'reports/li_accuracy_study/markov_global_test_predictions.csv')
    check=data.merge(old[['game_pk','at_bat_index','reference_validated','predicted_li','predicted_wpa']],
                     on=['game_pk','at_bat_index'],how='inner',validate='one_to_one')
    known=check.loc[check.reference_validated]
    li_error=float(abs(known.diagnostic_start_li-known.predicted_li).max())
    pairs=known.dropna(subset=['estimated_batter_wpa','predicted_wpa'])
    wpa_error=float(abs(pairs.estimated_batter_wpa-pairs.predicted_wpa).max())
    if li_error>1e-12 or wpa_error>1e-12 or len(pairs)!=4758:raise ValueError('Frozen heldout results changed')
    data.to_parquet(DATA,index=False);excluded.to_csv(OUT/'excluded_observations.csv',index=False)
    bygame=data.groupby('game_pk').agg(official_pa=('batter_id','size'),eligible_pa=('metric_eligible','sum'))
    bygame['excluded_pa']=bygame.official_pa-bygame.eligible_pa;bygame['excluded_fraction']=bygame.excluded_pa/bygame.official_pa
    bygame.to_csv(OUT/'coverage_by_game.csv')
    data.assign(excluded=~data.metric_eligible).groupby('result').agg(official_pa=('batter_id','size'),excluded_pa=('excluded','sum')).to_csv(OUT/'coverage_by_event.csv')
    excluded.groupby(['exclusion_reason','result']).size().rename('pa').to_csv(OUT/'exclusion_reasons.csv')
    quality={'generated_at_utc':utc_now(),'official_games':len(manifest),'official_pa':len(data),
        'metric_eligible_pa':int(data.metric_eligible.sum()),'excluded_pa':len(excluded),
        'coverage_fraction':float(data.metric_eligible.mean()),'network_requests':client.network_requests,
        'derived_cache_hits':hits,'known_reference_unresolved_quarantined':int(data.known_reference_unresolved.sum()),
        'heldout_li_max_change':li_error,'heldout_wpa_max_change':wpa_error,'model_refitted':False,
        'roles':data.groupby('model_role').agg(pa=('batter_id','size'),eligible=('metric_eligible','sum')).to_dict('index'),
        'model_inputs':'Saved global Markov kernel, saved calibration and saved LI normalizers; no application outcomes fit.'}
    atomic_json(OUT/'application_quality.json',quality)
    atomic_json(OUT/'application_provenance.json',{'generated_at_utc':utc_now(),'model_method_hash':method_hash,
        'game_receipts':receipts,'output_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
        'formula_sha256':hashlib.sha256((ROOT/'config/research_v1.json').read_bytes()).hexdigest()})
    print(json.dumps(quality,indent=2),flush=True)
    return data


if __name__=='__main__':apply()
