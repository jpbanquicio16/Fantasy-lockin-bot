import numpy as np
import pytest
from lockin_core.policy.exact_dp import ToyWeek, ToyPlayer, ToyGame, toy_problem, toy_scenarios, ExactDP, hindsight
from lockin_core.policy.stopping import premium_sd
from lockin_core.sim.engine import simulate
from lockin_core.sim.problem import Action, PolicyConfig, RootEntry
from lockin_core.state.information import InformationView, FutureInformationError
from lockin_core.state.legality import is_legal
from helpers import small_model


def week(final=10, scratch=0):
    return ToyWeek(('G',), (ToyPlayer(('PG',), (ToyGame(0,0,0,(100,),(1.,)),
                                             ToyGame(1,0,scratch,(final,),(1.,)))),), days=2)


@pytest.mark.parametrize('score,scratch,expected', [(10,0,10), (-10,0,-10), (10,1,0)])
def test_final_fallback_never_uses_earlier_game(score,scratch,expected):
    wk = week(score,scratch); prob = toy_problem(wk); sc = toy_scenarios(wk,20,1)
    result = simulate(prob,sc,policy=PolicyConfig(lock_rule='never', seat_by='highest_projection')).total
    np.testing.assert_array_equal(result, expected)


def test_benched_final_scores_zero():
    wk = ToyWeek(('G',), (ToyPlayer(('PG',), (ToyGame(0,0,0,(70,),(1.,)),)),), days=1)
    prob=toy_problem(wk); sc=toy_scenarios(wk,5,1)
    np.testing.assert_array_equal(simulate(prob,sc,Action(lineup=())).total, 0)


def test_future_information_is_unreadable():
    wk=week(); prob=toy_problem(wk); sc=toy_scenarios(wk,5,1); view=InformationView(prob,sc)
    with pytest.raises(FutureInformationError): view.final_score(0,0)
    with pytest.raises(FutureInformationError): view.status(0,0)
    view.advance(prob.players[0].tips[0]); assert not view.status(0,0).any()
    with pytest.raises(FutureInformationError): view.final_score(0,0)
    view.advance(prob.players[0].ends[0]); assert (view.final_score(0,0)==100).all()
    with pytest.raises(FutureInformationError): view.final_score(0,1)


@pytest.mark.parametrize('action', [Action(locks=((0,0),(0,0))), Action(locks=((0,-1),)),
    Action(locks=((20,0),)), Action(locks=((0,0),),passes=(0,)), Action(lineup=((0,0),(0,0)))])
def test_invalid_actions_rejected_at_simulator_boundary(action):
    wm=small_model(); assert not is_legal(wm.me,action)
    sc=wm.side_scenarios(wm.scenarios(3,'test'),wm.me)
    with pytest.raises(ValueError,match='Illegal'): simulate(wm.me,sc,action)


def test_locked_player_cannot_contribute_twice():
    wm=small_model(); sc=wm.side_scenarios(wm.scenarios(50,'test'),wm.me)
    np.testing.assert_array_equal(simulate(wm.me,sc,Action(locks=((0,0),))).total,60)


@pytest.mark.parametrize('games,premium', [(2,.399),(3,.630),(4,.790)])
def test_one_player_closed_form(games,premium):
    assert premium_sd(games) == pytest.approx(premium, abs=.001)


def test_exact_dp_bounds_on_small_case():
    wk=week(); prob=toy_problem(wk); sc=toy_scenarios(wk,50,1)
    optimum=ExactDP(wk).value()
    assert simulate(prob,sc).total.mean() <= optimum+1e-8
    assert optimum <= hindsight(wk,sc).mean()+1e-8


def test_chronological_rollout_keeps_base_decision_when_values_tie():
    from lockin_core.policy.rollout import RolloutSim
    wk=ToyWeek(('G',),(ToyPlayer(('PG',),(ToyGame(0,0,0,(10,40),(.5,.5)),
                                               ToyGame(1,0,0,(25,),(1.,)))),),days=2)
    prob=toy_problem(wk); actual=toy_scenarios(wk,8,1)
    rollout=RolloutSim(prob,lambda n,seed: toy_scenarios(wk,n,seed),scenarios=50)
    np.testing.assert_array_equal(rollout.run(actual).total,np.maximum(actual.units[:,0,0],25))
