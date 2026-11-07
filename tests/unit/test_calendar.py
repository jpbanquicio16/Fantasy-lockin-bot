from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import pytest

from lockin_core.config import UTC
from lockin_core.schedule.calendar import week_bounds


@pytest.mark.parametrize('zone,local,hours', [
    ('America/New_York', datetime(2026,11,1,1,30), 1),
    ('Australia/Adelaide', datetime(2027,4,4,2,30), 1)])
def test_ambiguous_times_are_distinct_instants(zone, local, hours):
    z=ZoneInfo(zone)
    first=local.replace(tzinfo=z,fold=0).astimezone(UTC)
    second=local.replace(tzinfo=z,fold=1).astimezone(UTC)
    assert second-first==timedelta(hours=hours)


def test_late_sunday_belongs_to_previous_et_week():
    tip=datetime(2026,11,9,3,30,tzinfo=UTC)
    et=tip.astimezone(ZoneInfo('America/New_York'))
    assert tip.astimezone(ZoneInfo('Australia/Adelaide')).weekday()==0
    start,end=week_bounds([et.date()])
    assert start.astimezone(ZoneInfo('America/New_York')).date()==date(2026,11,2)
    assert start<=tip<end


@pytest.mark.parametrize('day,hours', [(date(2026,11,1),169),(date(2027,3,14),167)])
def test_et_week_duration_across_dst(day,hours):
    start,end=week_bounds([day]); assert (end-start).total_seconds()==hours*3600
