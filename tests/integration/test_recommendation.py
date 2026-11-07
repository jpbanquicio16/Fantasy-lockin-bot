from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from helpers import small_model
from lockin_core.config import Paths
from lockin_core.state.store import Store
from lineup_lock.recommend.engine import Recommender
from lineup_lock.replay.artifacts import save, load, audit, canonical
from lineup_lock.replay.shadow import report


def test_recommendation_reproduces_from_safe_snapshot(tmp_path):
    wm=small_model(); paths=Paths(root=tmp_path).ensure()
    rec=Recommender(wm,wm.scfg).run(fixed_scenarios=50)
    save(paths,rec,wm)
    restored,stored=load(paths,rec['rec_id'])
    np.testing.assert_array_equal(restored.me.tables.W,wm.me.tables.W)
    assert restored.ctx.calendar.week_end==wm.ctx.calendar.week_end
    assert stored==rec
    assert audit(paths,rec['rec_id'])
    assert rec['summary']['p_win'] is None
    assert rec['entries'][0]['lock_slot_index']==0


def test_win_requires_confirmed_opponent_total():
    wm=small_model()
    with pytest.raises(ValueError,match='opponent'):
        Recommender(wm,wm.scfg).run('win')


def test_stale_report_cannot_recommend_wait():
    wm=small_model(); wm.ctx.report_published_at=wm.state.t-timedelta(hours=4)
    rec=Recommender(wm,wm.scfg).run(fixed_scenarios=50)
    assert all(c['advice']!='wait' for c in rec['entries'])


def test_shadow_does_not_claim_unobserved_gain(tmp_path):
    store=Store(path=tmp_path/'app.sqlite')
    assert 'No claim of improvement' in report(store)
    wm=small_model(); rec=Recommender(wm,wm.scfg).run(fixed_scenarios=50)
    store.save_recommendation({'rec_id':rec['rec_id'],'created_at':rec['created_at'],'result':rec})
    assert 'Pending: 1' in report(store)
    store.add_outcome(rec['rec_id'],{},True,100,{'recommended':110})
    assert '+5.00 FP across 1 weeks' in report(store)


def test_app_setup_empty_workspace(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('LOCKIN_ROOT',str(tmp_path))
    app=Path(__file__).parents[2]/'src/lineup_lock/app/streamlit_app.py'
    at=AppTest.from_file(str(app)).run(timeout=30)
    assert not at.exception
    assert any('Set your league ID' in x.value for x in at.info)


def test_app_displays_recommendation(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    from lockin_core.config import now_utc
    paths=Paths(root=tmp_path).ensure(); paths.configs.mkdir()
    (paths.configs/'league.yaml').write_text('league_id: test\nmy_roster_id: 1\n')
    wm=small_model(now_utc()); rec=Recommender(wm,wm.scfg).run(fixed_scenarios=50)
    Store(paths).save_recommendation({'rec_id':rec['rec_id'],'created_at':rec['created_at'],
                                     'expires_at':rec['expires_at'],'result':rec})
    monkeypatch.setenv('LOCKIN_ROOT',str(tmp_path))
    app=Path(__file__).parents[2]/'src/lineup_lock/app/streamlit_app.py'
    at=AppTest.from_file(str(app)).run(timeout=30)
    assert not at.exception
    assert any(x.label=='Expected week points' for x in at.metric)


def test_model_versions_can_roll_back(tmp_path):
    from lockin_core.models.fit import save as save_model, load as load_model, activate_version
    wm=small_model(); paths=Paths(root=tmp_path).ensure()
    save_model(paths,wm.params,wm.lib)
    second=wm.params.model_copy(update={'version':'second'})
    save_model(paths,second,wm.lib,activate=False)
    assert load_model(paths)[0].version=='test'
    activate_version(paths,'second')
    assert load_model(paths)[0].version=='second'
    activate_version(paths,'test')
    assert load_model(paths)[0].version=='test'
