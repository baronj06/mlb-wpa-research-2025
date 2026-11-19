"""HTTP/checkpoint tests use fake responses; no external data are invented."""

import json
from pathlib import Path

import pytest
import requests

from mlb_wpa.collection import CachedClient, GameCollector, ensure_scope, manifest_from_schedule, season_plan


def response(url, data=b"payload", status=200, headers=None):
    r = requests.Response()
    r.url, r.status_code, r._content = url, status, data
    r.headers.update(headers or {})
    return r


class FakeSession:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


URL = "https://statsapi.mlb.com/api/v1/game/1/winProbability"


def test_cache_reuse_offline_and_tamper_detection(tmp_path):
    session = FakeSession([response(URL)])
    client = CachedClient(tmp_path, session=session, sleep=lambda _: None)
    content, first = client.get(URL)
    assert client.get(URL) == (content, first)
    assert len(session.calls) == 1
    assert CachedClient(tmp_path, offline=True).get(URL)[0] == content
    (tmp_path / "blobs" / first["sha256"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        client.get(URL)
    assert len(session.calls) == 1  # No silent replacement with a network response.


def test_offline_miss_never_uses_network(tmp_path):
    s = FakeSession([])
    with pytest.raises(FileNotFoundError):
        CachedClient(tmp_path, session=s, offline=True).get(URL)
    assert not s.calls


@pytest.mark.parametrize("first", [429, 503])
def test_transient_status_is_retried_and_recorded(tmp_path, first):
    delays = []
    s = FakeSession([response(URL, status=first, headers={"Retry-After": "100"}), response(URL)])
    _, receipt = CachedClient(tmp_path, session=s, sleep=delays.append).get(URL)
    assert [a["status"] for a in receipt["attempts"]] == [first, 200]
    assert 30 in delays  # Retry-After is bounded; no long blocking wait.


def test_timeout_retries_without_disabling_tls(tmp_path):
    s = FakeSession([requests.Timeout("synthetic timeout"), response(URL)])
    _, receipt = CachedClient(tmp_path, session=s, sleep=lambda _: None).get(URL)
    assert receipt["tls_verification"] is True
    assert receipt["attempts"][0]["error"] == "Timeout"
    assert len(s.calls) == 2


@pytest.mark.parametrize("status", [403, 404])
def test_permanent_http_error_no_retry_and_failure_receipt(tmp_path, status):
    s = FakeSession([response(URL, status=status)])
    with pytest.raises(requests.HTTPError):
        CachedClient(tmp_path, session=s, sleep=lambda _: None).get(URL)
    assert len(s.calls) == 1
    assert list((tmp_path / "failures").glob("*.json"))
    assert not list((tmp_path / "requests").glob("*.json"))


def test_retry_exhaustion_is_bounded(tmp_path):
    s = FakeSession([response(URL, status=503) for _ in range(3)])
    with pytest.raises(requests.HTTPError):
        CachedClient(tmp_path, session=s, sleep=lambda _: None, retries=2).get(URL)
    assert len(s.calls) == 3


def test_refresh_preserves_all_observation_manifests(tmp_path):
    s = FakeSession([response(URL), response(URL), response(URL, b"revised")])
    c = CachedClient(tmp_path, session=s, sleep=lambda _: None)
    _, original = c.get(URL)
    c.get(URL, refresh=True)
    _, revised = c.get(URL, refresh=True)
    assert len(list((tmp_path / "manifests").glob("*.json"))) == 3
    assert (tmp_path / original["snapshot_manifest"]).exists()
    assert original["sha256"] != revised["sha256"]
    assert (tmp_path / "blobs" / original["sha256"]).read_bytes() == b"payload"


@pytest.mark.parametrize("url", ["http://statsapi.mlb.com/a", "https://example.org/a"])
def test_non_allowlisted_sources_rejected_before_http(tmp_path, url):
    s = FakeSession([])
    with pytest.raises(ValueError):
        CachedClient(tmp_path, session=s).get(url)
    assert not s.calls


def sample_manifest():
    return {"scope": "validation_sample", "games": [{"game_pk": 1, "season": 2025,
            "date": "2025-04-01", "home_team_name": "Chicago Cubs", "dh": 0}]}


def metadata_bytes():
    return json.dumps({"gameData": {"game": {"pk": 1, "season": "2025", "type": "R"},
                                   "datetime": {"officialDate": "2025-04-01"},
                                   "teams": {"home": {"teamName": "Cubs"}},
                                   "status": {"abstractGameState": "Final"}}}).encode()


def test_checkpoint_resume_reuses_successful_sources(tmp_path):
    graph = "https://www.fangraphs.com/scores/game-graphs?date=2025-04-01&hometeamid=17&dh=0"
    s = FakeSession([response(URL, metadata_bytes()), response(URL, b"[]"), response(graph, b"synthetic box"),
                     response("https://www.fangraphs.com/scores/play-log", status=503)])
    # Explicit synthetic transport: no real provider request is made.
    c = CachedClient(tmp_path / "cache", session=s, retries=0, sleep=lambda _: None, network_authorizer=lambda _:None)
    path = tmp_path / "checkpoint.json"
    failed = GameCollector(c, path).collect(sample_manifest())
    assert failed["games"]["1"]["collection_status"] == "failed"
    assert len(failed["games"]["1"]["sources"]) == 3
    resumed = FakeSession([response("https://www.fangraphs.com/scores/play-log", b"synthetic log")])
    done = GameCollector(CachedClient(c.root, session=resumed, sleep=lambda _: None, network_authorizer=lambda _:None), path).collect(sample_manifest())
    assert len(resumed.calls) == 1
    assert done["games"]["1"]["collection_status"] == "raw_complete"
    assert done["games"]["1"]["validation_status"] == "not_run"
    assert "error" not in done["games"]["1"]


def test_refreshed_source_invalidates_prior_validation(tmp_path):
    graph = "https://www.fangraphs.com/scores/game-graphs?hometeamid=17"
    replies = [response(URL, metadata_bytes()), response(URL, b"[]"), response(graph),
               response("https://www.fangraphs.com/scores/play-log")]
    c = CachedClient(tmp_path / "cache", session=FakeSession(list(replies)), sleep=lambda _: None, network_authorizer=lambda _:None)
    path = tmp_path / "checkpoint.json"
    state = GameCollector(c, path).collect(sample_manifest())
    state["games"]["1"]["validation_status"] = "passed"
    path.write_text(json.dumps(state))
    changed = list(replies)
    changed[1] = response(URL, b"[{\"source_revision\":true}]")
    new = GameCollector(CachedClient(c.root, session=FakeSession(changed), sleep=lambda _: None, network_authorizer=lambda _:None), path).collect(sample_manifest(), refresh=True)
    assert new["games"]["1"]["validation_status"] == "not_run"


def test_season_plan_and_manifest_are_pure_and_execution_gated():
    plan = season_plan()
    assert plan["games"] == [] and not plan["execution_authorized"]
    with pytest.raises(PermissionError):
        ensure_scope(plan)
    m = sample_manifest()
    m["games"] = [{**m["games"][0], "game_pk": i} for i in range(1, 12)]
    with pytest.raises(PermissionError):
        ensure_scope(m)


@pytest.mark.parametrize("field,value", [("game_pk",True),("game_pk",-1),("game_pk","1"),("season",2024),("date","2024-04-01")])
def test_invalid_collection_scope_rejected(field, value):
    m = sample_manifest()
    m["games"][0][field] = value
    with pytest.raises(ValueError):
        ensure_scope(m)


def test_supplied_schedule_deduplicates_and_flags_doubleheaders():
    game = {"gamePk": 1, "season": "2025", "gameType": "R", "doubleHeader": "Y",
            "officialDate": "2025-04-01", "teams": {"home": {"team": {"name": "Cubs"}}, "away": {"team": {"name": "Dodgers"}}},
            "status": {"abstractGameState": "Final"}}
    plan = manifest_from_schedule({"dates": [{"games": [game, game, {**game, "gamePk": 2, "gameType": "S"}]}]})
    assert len(plan["games"]) == 1
    assert plan["games"][0]["route_requires_review"] is True
    assert plan["games"][0]["dh"] is None
    assert not plan["execution_authorized"]


def test_unverified_doubleheader_blocked_without_network(tmp_path):
    m = sample_manifest()
    m["games"][0]["route_requires_review"] = True
    s = FakeSession([])
    state = GameCollector(CachedClient(tmp_path / "cache", session=s), tmp_path / "state.json").collect(m)
    assert not s.calls
    assert state["games"]["1"]["collection_status"] == "failed"
