from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from helpers import small_model
from lockin_core.config import Paths, AlertsConfig
from lockin_core.state.store import Store
from lineup_lock.events.service import Service


class Notifications:
    channel = 'test'
    def __init__(self): self.messages = []; self.ok = True
    def send(self, title, body, when):
        self.messages.append((title, body))
        return self.ok


def service(tmp_path):
    wm=small_model(); paths=Paths(root=tmp_path).ensure(); store=Store(paths); notifier=Notifications()
    svc=Service(paths, wm.ctx.cfg, wm.mcfg, wm.scfg, AlertsConfig(),
                client=SimpleNamespace(), espn=SimpleNamespace(), store=store, notifier=notifier)
    return svc, wm.state.t


def deadline(svc, now):
    svc.store.upsert_deadline('entry',now+timedelta(minutes=8),
                              {'player':'P0','score':30,'default':'lock in G'},None,[120,30,10])


def test_alert_uses_urgent_offset_once_across_restart(tmp_path):
    svc,now=service(tmp_path); deadline(svc,now)
    assert svc.alerts(now)==1
    assert svc.store.alerts()[0]['kind']=='T-10'
    assert svc.alerts(now+timedelta(minutes=1))==0
    assert 'stored default' in svc.notifier.messages[0][1]


def test_failed_delivery_retried(tmp_path):
    svc,now=service(tmp_path); deadline(svc,now); svc.notifier.ok=False
    assert svc.alerts(now)==0
    svc.notifier.ok=True
    assert svc.alerts(now+timedelta(minutes=1))==1


def test_alerts_survive_all_upstream_failures(tmp_path):
    svc,now=service(tmp_path); deadline(svc,now)
    with patch.object(svc,'calendar',side_effect=RuntimeError('network down')), patch(
            'lineup_lock.events.service.ingest',side_effect=RuntimeError('network down')):
        result=svc.tick(now)
    assert svc.notifier.messages
    assert result.errors
    assert svc.store.get('last_tick')==now.isoformat()


def test_missed_deadline_is_counted_and_reported(tmp_path):
    svc,now=service(tmp_path); deadline(svc,now)
    svc.alerts(now+timedelta(minutes=9))
    assert len(svc.store.deadlines('missed'))==1
    assert svc.notifier.messages[0][0]=='Missed deadline alert'


def test_rescheduled_deadline_gets_new_alert(tmp_path):
    svc,now=service(tmp_path); deadline(svc,now); svc.alerts(now)
    svc.store.upsert_deadline('entry',now+timedelta(minutes=18),{'player':'P0','default':'pass'},None,[10])
    assert svc.alerts(now+timedelta(minutes=10))==1
    assert len(svc.store.alerts())==2


def test_expired_advice_not_used(tmp_path):
    svc,now=service(tmp_path); deadline(svc,now)
    rec={'rec_id':'r','created_at':now.isoformat(),'expires_at':(now-timedelta(seconds=1)).isoformat(),
         'result':{'entries':[{'entry_id':'entry','advice':'wait'}]}}
    svc.store.save_recommendation(rec)
    svc.alerts(now)
    assert 'stored default' in svc.notifier.messages[0][1]


def test_invalidated_advice_hidden_and_revision_incremented(tmp_path):
    svc,now=service(tmp_path)
    svc.store.save_recommendation({'rec_id':'r','created_at':now.isoformat(),'result':{}})
    svc.store.invalidate('manual lock')
    assert svc.store.latest_recommendation() is None
    assert svc.store.get('force_recommend')
    assert svc.store.get('input_revision')==1


def test_backup_round_trip(tmp_path):
    svc,now=service(tmp_path); deadline(svc,now)
    path=svc.store.backup(tmp_path/'backup.sqlite')
    assert Store(path=path).deadlines()[0]['entry_id']=='entry'
    assert svc.store.export_history(tmp_path/'history.json.gz').exists()
