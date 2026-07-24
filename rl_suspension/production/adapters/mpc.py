"""Generic wrapper for an existing private MPC solver."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from rl_suspension.production.contracts import (
    DEFAULT_ACTION_SCHEMA,
    ActionSchema,
    MpcSolveResult,
    ObservationV1,
    Scenario,
    SolverDiagnostics,
)


@dataclass
class CallableMpcAdapter:
    """Adapt reset/solve callables while enforcing the production contract."""

    solve_fn: Callable[[ObservationV1, bytes], MpcSolveResult | dict]
    reset_fn: Callable[[Scenario, bytes], None] | None = None
    action_schema: ActionSchema = DEFAULT_ACTION_SCHEMA
    timeout_ms: float = 1000.0
    name: str = "private_mpc"

    def reset(self, scenario: Scenario, simulator_snapshot: bytes) -> None:
        if self.reset_fn is not None:
            self.reset_fn(scenario, simulator_snapshot)

    def solve(
        self,
        observation: ObservationV1,
        simulator_snapshot: bytes,
    ) -> MpcSolveResult:
        started = time.perf_counter()
        raw = self.solve_fn(observation, simulator_snapshot)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        result = raw if isinstance(raw, MpcSolveResult) else self._from_mapping(raw)
        action = self.action_schema.validate(result.action_12d, bounded=False)
        timeout = elapsed_ms > self.timeout_ms
        within_bounds = np.all(action >= np.asarray(self.action_schema.minimum)) and np.all(
            action <= np.asarray(self.action_schema.maximum)
        )
        diagnostics = SolverDiagnostics(
            status=result.diagnostics.status,
            objective=result.diagnostics.objective,
            iterations=result.diagnostics.iterations,
            solve_time_ms=elapsed_ms,
            feasibility_margin=result.diagnostics.feasibility_margin,
            fallback=result.diagnostics.fallback,
            timeout=timeout,
            extra=dict(result.diagnostics.extra),
        )
        return MpcSolveResult(
            action_12d=np.clip(
                action,
                np.asarray(self.action_schema.minimum),
                np.asarray(self.action_schema.maximum),
            ),
            valid=bool(result.valid and within_bounds and not timeout),
            diagnostics=diagnostics,
            horizon_summary=dict(result.horizon_summary),
        )

    @staticmethod
    def _from_mapping(raw: dict) -> MpcSolveResult:
        diagnostics = raw.get("diagnostics", {})
        return MpcSolveResult(
            action_12d=np.asarray(raw["action_12d"], dtype=np.float64),
            valid=bool(raw.get("valid", True)),
            diagnostics=SolverDiagnostics(
                status=str(diagnostics.get("status", "unknown")),
                objective=float(diagnostics.get("objective", np.inf)),
                iterations=int(diagnostics.get("iterations", 0)),
                solve_time_ms=float(diagnostics.get("solve_time_ms", 0.0)),
                feasibility_margin=float(
                    diagnostics.get("feasibility_margin", -np.inf)
                ),
                fallback=bool(diagnostics.get("fallback", False)),
                timeout=bool(diagnostics.get("timeout", False)),
                extra=dict(diagnostics.get("extra", {})),
            ),
            horizon_summary=dict(raw.get("horizon_summary", {})),
        )
