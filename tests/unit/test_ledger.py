from datetime import timedelta
from unittest.mock import patch
import pytest

from helpers import small_model
from lockin_core.state.store import Store
from lockin_core.state.ledger import resolve_tipoffs, starters_at, tip_batch_id
from lockin_core.state.build import derive_side, WeekGameScore
from lockin_core.state.manual import record_entry, score_units, confirm_tip


def side(store, wm, starters=None, shown=None):
    return derive_side(1, "me", ["G"], starters or ["0"], ["0", "1"], [], shown or [30],
                       wm.ctx, store, "test", 3, wm.state.t, 2)


def test_missing_tip_never_inferred_from_points(tmp_path):
    wm = small_model(); store = Store(path=tmp_path/'app.sqlite')
    wm.ctx.scores = {"0": [WeekGameScore("old", "AAA", "played", 60, 1)]}
    result = side(store, wm)
    assert not result.entries
    assert result.notes[0].startswith('eligibility_uncertain:')


@pytest.mark.parametrize('started,eligible', [(["0"], True), (["1"], False)])
def test_eligibility_requires_start_at_tip(tmp_path, started, eligible):
    wm = small_model(); store = Store(path=tmp_path/'app.sqlite'); tip = wm.ctx.calendar.games[0].tip_at
    store.save_tipoff(tip_batch_id(tip), 'test', 1, tip, None, None, started, 'observed', recorded_at=tip)
    wm.ctx.scores = {"0": [WeekGameScore("old", "AAA", "played", 60, 1)]}
    assert bool(side(store, wm).entries) == eligible


def test_no_retroactive_manual_correction(tmp_path):
    wm = small_model(); store = Store(path=tmp_path/'app.sqlite'); tip = wm.ctx.calendar.games[0].tip_at
    store.save_tipoff(tip_batch_id(tip), 'test', 1, tip, None, None, ['0'], 'observed', recorded_at=tip)
    store.save_tipoff(tip_batch_id(tip), 'test', 1, tip, None, None, ['1'], 'confirmed_manual', recorded_at=wm.state.t)
    assert starters_at(store, 1, tip, as_of=tip+timedelta(hours=1))[0] == ['0']
    assert starters_at(store, 1, tip, as_of=wm.state.t)[0] == ['1']


def test_lineup_churn_loses_entry(tmp_path):
    wm = small_model(); store = Store(path=tmp_path/'app.sqlite'); tip = wm.ctx.calendar.games[0].tip_at
    store.save_tipoff(tip_batch_id(tip), 'test', 1, tip, None, None, ['0'], 'observed', recorded_at=tip)
    store.add_observation('test', 1, tip+timedelta(minutes=5), ['1'], ['0'], [])
    wm.ctx.scores = {"0": [WeekGameScore("old", "AAA", "played", 60, 1)]}
    assert not side(store, wm).entries


@pytest.mark.parametrize('post,expected', [(['0'], 'observed'), (['1'], 'uncertain')])
def test_tip_resolution(tmp_path, post, expected):
    wm = small_model(); store = Store(path=tmp_path/'app.sqlite'); tip = wm.ctx.calendar.games[0].tip_at
    store.add_observation('test', 1, tip-timedelta(minutes=2), ['0'], [], [])
    store.add_observation('test', 1, tip+timedelta(minutes=2), post, [], [])
    resolve_tipoffs(store, 'test', 1, wm.ctx.calendar, tip+timedelta(minutes=3))
    assert starters_at(store, 1, tip)[1] == expected


def test_manual_slot_is_authoritative_and_duplicate_slot_rejected(tmp_path):
    wm = small_model(); store = Store(path=tmp_path/'app.sqlite')
    with patch('lockin_core.state.store.now_utc', return_value=wm.state.t-timedelta(minutes=1)):
        store.record_lock('test', 3, 1, 0, 'G', '0', 'old', 60)
        with pytest.raises(ValueError, match='slot'):
            store.record_lock('test', 3, 1, 0, 'G', '1', 'other', 50)
    assert side(store, wm, ['1']).locked[0].player_id == '0'


@pytest.mark.parametrize('offset,allowed', [(-1, True), (0, False), (1, False)])
def test_deadline_boundary(tmp_path, offset, allowed):
    wm = small_model(); store = Store(path=tmp_path/'app.sqlite'); entry = wm.state.me.entries[0]
    with patch('lockin_core.state.manual.now_utc', return_value=entry.deadline_at+timedelta(seconds=offset)):
        if allowed:
            record_entry(store, wm.state, entry.entry_id, 'lock', 0)
        else:
            with pytest.raises(ValueError, match='deadline'):
                record_entry(store, wm.state, entry.entry_id, 'lock', 0)


def test_scores_not_silently_rounded():
    assert score_units(-1.5, 2) == -3
    for value in [1.25, float('nan'), float('inf')]:
        with pytest.raises(ValueError): score_units(value, 2)


def test_bad_tip_time_rejected(tmp_path):
    wm = small_model(); store = Store(path=tmp_path/'app.sqlite')
    with pytest.raises(ValueError, match='timezone'):
        confirm_tip(store, 'test', 1, wm.state.t.replace(tzinfo=None), ['0'])
