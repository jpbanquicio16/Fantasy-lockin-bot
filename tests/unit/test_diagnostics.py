import pytest
from lockin_core.models.evaluate import availability, distributions
from lineup_lock.replay.shadow import week_interval
from lineup_lock.app.access import validate_bind_address


def test_forecast_metrics():
    assert availability([1,0],[1,0])['brier']==0
    assert distributions([[10,10],[20,20]],[10,20])['crps']==0
    assert distributions([[10,10]],[15])['crps']==5


def test_block_interval_counts_weeks_not_decisions():
    result=week_interval({(2026,1):[1]*100,(2026,2):[3]})
    assert result[0]==2
    assert result[3]==2


@pytest.mark.parametrize('address', ['127.0.0.1','::1','100.101.102.103','fd7a:115c:a1e0::1'])
def test_private_binding(address): validate_bind_address(address)


@pytest.mark.parametrize('address', ['0.0.0.0','::','8.8.8.8','192.168.1.2'])
def test_public_binding_rejected(address):
    with pytest.raises(ValueError): validate_bind_address(address)
