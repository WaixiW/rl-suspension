"""Shared rollout + scenario evaluation helpers used by the validation suite.

A single rollout function drives any controller kind over one episode and
returns the EpisodeMetrics plus optional time-series for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .env import SuspensionEnv
from .metrics import EpisodeMetrics
from .mpc import MPCTeacher


@dataclass
class Trace:
    t: np.ndarray
    heave_acc: np.ndarray
    susp_defl: np.ndarray          # (T,4)
    command: np.ndarray            # (T,12) normalized
    safety: np.ndarray             # (T,) intervention flags


def rollout(
    env: SuspensionEnv,
    kind: str,
    controller,
    record_trace: bool = False,
) -> Tuple[EpisodeMetrics, Optional[Trace]]:
    """Run one episode. `kind` in {passive, skyhook, mpc, rl}."""
    out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    if hasattr(controller, "reset"):
        controller.reset()
    # MPC must be rebuilt against the (possibly randomized) car for this episode.
    if kind == "mpc":
        controller = MPCTeacher(env.car, env.cfg.actuator, env.cfg.sim.control_dt)

    ts, ha, sd, cm, sf = [], [], [], [], []
    dt = env.cfg.sim.control_dt
    for k in range(env._max_control_steps()):
        if kind in ("passive", "skyhook"):
            inp = env.controller_inputs()
            cmd = controller.act(inp["est"])
            res = env.apply_command(cmd)
        elif kind == "mpc":
            inp = env.controller_inputs()
            cmd, _ = controller.act(inp["x_hat"], inp["rel_vel"], inp["road_future"])
            res = env.apply_command(cmd)
        elif kind == "rl":
            cmd = controller.act_obs(obs)
            res = env.apply_command(cmd)
        else:
            raise ValueError(f"unknown controller kind: {kind}")
        obs = res.obs
        if record_trace:
            o = res.info["outputs"]
            ts.append(k * dt)
            ha.append(o.heave_acc)
            sd.append(o.susp_defl.copy())
            cm.append(env.prev_action.copy())
            sf.append(float(res.info["safety_intervened"]))
        if res.terminated:
            break
    metrics = env.episode_metrics()
    trace = None
    if record_trace and ts:
        trace = Trace(
            t=np.array(ts),
            heave_acc=np.array(ha),
            susp_defl=np.array(sd),
            command=np.array(cm),
            safety=np.array(sf),
        )
    return metrics, trace


def evaluate_scenarios(
    make_env,
    controllers: Dict[str, Tuple[str, object]],
    scenarios: List[str],
    n_seeds: int = 1,
    base_seed: int = 1000,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return results[scenario][controller] = {metric: mean, metric_std: std}."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scen in scenarios:
        results[scen] = {}
        for cname, (kind, ctrl) in controllers.items():
            per_seed: List[EpisodeMetrics] = []
            for s in range(n_seeds):
                env = make_env(seed=base_seed + s)
                env.set_scenario(scen)
                m, _ = rollout(env, kind, ctrl)
                per_seed.append(m)
            results[scen][cname] = _aggregate(per_seed)
    return results


def _aggregate(metrics_list: List[EpisodeMetrics]) -> Dict[str, float]:
    keys = metrics_list[0].to_dict().keys()
    agg: Dict[str, float] = {}
    for k in keys:
        vals = np.array([m.to_dict()[k] for m in metrics_list], dtype=float)
        agg[k] = float(np.mean(vals))
        agg[k + "_std"] = float(np.std(vals))
    return agg
