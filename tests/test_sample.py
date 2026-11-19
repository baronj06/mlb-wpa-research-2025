import copy
import json
from pathlib import Path

import pytest

from mlb_wpa.sample import normalize


@pytest.fixture
def payload():
    return json.loads((Path(__file__).parents[1] / "data/raw/745444_win_probability.json").read_text())


def test_real_sample(payload):
    rows = normalize(payload)
    assert len(rows) == 79
    assert [r["at_bat_index"] for r in rows if r["li"] is None] == [0]
    assert len({(r["game_pk"], r["at_bat_index"]) for r in rows}) == 79
    assert rows[0]["batting_team_event_wpa"] == pytest.approx(0.036)
    assert rows[-1]["home_wp_after"] == 0


def test_sign_and_units(payload):
    for raw, row in zip(payload, normalize(payload), strict=True):
        sign = -1 if raw["about"]["halfInning"] == "top" else 1
        assert row["batting_team_event_wpa"] == pytest.approx(sign * raw["homeTeamWinProbabilityAdded"] / 100)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "1.0", True])
def test_invalid_li(payload, value):
    payload[1]["leverageIndex"] = value
    with pytest.raises(ValueError):
        normalize(payload)


def test_zero_li_preserved(payload):
    payload[1]["leverageIndex"] = 0
    assert normalize(payload)[1]["li"] == 0


def test_duplicate_rejected(payload):
    payload.append(copy.deepcopy(payload[0]))
    with pytest.raises(ValueError, match="duplicate"):
        normalize(payload)


def test_probability_complement(payload):
    payload[0]["awayTeamWinProbability"] = 10
    with pytest.raises(ValueError, match="complement"):
        normalize(payload)


def test_non_pa_and_incomplete_excluded(payload):
    payload[0]["credits"] = []
    payload[1]["about"]["isComplete"] = False
    assert len(normalize(payload)) == 77


@pytest.mark.parametrize("value", [None, 101, float("nan")])
def test_invalid_wpa(payload, value):
    payload[0]["homeTeamWinProbabilityAdded"] = value
    with pytest.raises(ValueError):
        normalize(payload)
