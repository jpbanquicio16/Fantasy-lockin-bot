"""The simulator's time-gated interface to sampled outcomes."""
import numpy as np
from lockin_core.sim.problem import SideProblem, SideScenarios


class FutureInformationError(RuntimeError):
    pass


class InformationView:
    def __init__(self, problem: SideProblem, scenarios: SideScenarios):
        self._problem = problem
        self._scenarios = scenarios
        self.t = 0.0

    def advance(self, minute: float):
        if minute < self.t:
            raise ValueError("Information time cannot go backwards")
        self.t = minute

    def status(self, player: int, game: int):
        if self.t < self._problem.players[player].tips[game]:
            raise FutureInformationError("Availability is not yet revealed")
        return self._scenarios.out[:, player, game].copy()

    def final_score(self, player: int, game: int):
        if self.t < self._problem.players[player].ends[game]:
            raise FutureInformationError("Score is not final at this decision time")
        result = self._scenarios.units[:, player, game]
        if not np.isfinite(result).all():
            raise FutureInformationError("Non-finite score reached a final event")
        return result.copy()

    def scratched(self, player: int, game: int):
        if self.t < self._problem.players[player].ends[game]:
            raise FutureInformationError("Late scratch is not yet revealed")
        return self._scenarios.scr[:, player, game].copy()
