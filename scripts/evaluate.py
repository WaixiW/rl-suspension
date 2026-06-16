"""Staged validation suite for the active-suspension controllers.

Stages (mirrors the plan's validation roadmap):
  1. Nominal report  : per-scenario comfort/handling for passive/skyhook/mpc/rl.
  2. Monte Carlo     : randomized (domain-randomized) robustness, mean +/- std.
  3. Preview-error   : RL comfort vs increasing preview noise (perception gap).
  4. DR-width sweep  : RL graceful degradation vs randomization width (open risk).
  5. Safety filter   : harsh scenario with/without the command-level safety filter.
  6. Fallback table  : FallbackSupervisor decisions vs health signals.

Outputs runs/eval/{report.md, metrics.json, *.png}. RL/BC are included only if
their trained artifacts exist.

Usage:
    python scripts/evaluate.py --rl runs/ppo_main --bc runs/bc --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_suspension.config import default_config
from active_suspension.env import SuspensionEnv
from active_suspension.baselines import PassiveController, SkyhookController
from active_suspension.domain_randomization import DRConfig
from active_suspension.evaluation import rollout, evaluate_scenarios
from active_suspension.safety import FallbackSupervisor, FallbackMode

VAL_SCENARIOS = [
    "single_bump", "double_bump", "asymmetric_bump",
    "pothole", "speed_bump", "washboard", "iso_C", "iso_D",
]


def env_factory(randomize=False, dr_width=1.0, use_safety=False,
                preview_noise_scale=1.0, base_cfg=None):
    def make(seed=0):
        cfg = (base_cfg or default_config()).copy()
        cfg.preview.height_noise_std *= preview_noise_scale
        cfg.preview.dropout_prob = min(1.0, cfg.preview.dropout_prob * preview_noise_scale)
        env = SuspensionEnv(
            cfg,
            dr=DRConfig(enabled=randomize, width=dr_width),
            randomize=randomize,
            use_safety=use_safety,
            seed=seed,
        )
        return env
    return make


def load_controllers(rl_dir, bc_dir, actuator):
    ctrls = {
        "passive": ("passive", PassiveController(actuator)),
        "skyhook": ("skyhook", SkyhookController(actuator)),
        "mpc": ("mpc", None),
    }
    from active_suspension.policy import RLPolicy
    if rl_dir and os.path.exists(os.path.join(rl_dir, "policy.zip")):
        vn = os.path.join(rl_dir, "vecnormalize.pkl")
        ctrls["rl"] = ("rl", RLPolicy(os.path.join(rl_dir, "policy"),
                                      vn if os.path.exists(vn) else None, actuator))
    if bc_dir and os.path.exists(os.path.join(bc_dir, "policy.zip")):
        vn = os.path.join(bc_dir, "vecnormalize.pkl")
        ctrls["bc"] = ("rl", RLPolicy(os.path.join(bc_dir, "policy"),
                                      vn if os.path.exists(vn) else None, actuator))
    return ctrls


# ---------------------------------------------------------------- stages
def stage_nominal(ctrls):
    make = env_factory(randomize=False)
    return evaluate_scenarios(make, ctrls, VAL_SCENARIOS, n_seeds=1)


def stage_montecarlo(ctrls, seeds):
    make = env_factory(randomize=True, dr_width=1.0)
    sub = {k: v for k, v in ctrls.items() if k in ("passive", "mpc", "rl", "bc")}
    return evaluate_scenarios(make, sub, VAL_SCENARIOS, n_seeds=seeds)


def stage_preview_sweep(ctrls, seeds):
    if "rl" not in ctrls:
        return {}
    res = {}
    for scale in [0.0, 1.0, 3.0, 6.0]:
        make = env_factory(randomize=True, dr_width=1.0, preview_noise_scale=scale)
        vals = []
        for s in range(seeds):
            env = make(seed=3000 + s)
            env.set_scenario("iso_C")
            m, _ = rollout(env, "rl", ctrls["rl"][1])
            vals.append(m.comfort_wk_rms)
        res[scale] = (float(np.mean(vals)), float(np.std(vals)))
    return res


def stage_dr_sweep(ctrls, seeds):
    if "rl" not in ctrls:
        return {}
    res = {}
    for width in [0.0, 0.5, 1.0, 1.5, 2.0]:
        make = env_factory(randomize=width > 0, dr_width=max(width, 1e-6))
        comfort, viol = [], []
        for s in range(seeds):
            env = make(seed=4000 + s)
            env.set_scenario("iso_C")
            m, _ = rollout(env, "rl", ctrls["rl"][1])
            comfort.append(m.comfort_wk_rms)
            viol.append(m.travel_violations)
        res[width] = (float(np.mean(comfort)), float(np.mean(viol)))
    return res


def stage_safety(ctrls, seeds):
    if "rl" not in ctrls:
        return {}
    # harsh scenario: tall speed bump + DR
    out = {}
    for use_safety in (False, True):
        make = env_factory(randomize=True, dr_width=1.5, use_safety=use_safety)
        comfort, viol, interv = [], [], []
        for s in range(seeds):
            env = make(seed=5000 + s)
            env.set_scenario("speed_bump")
            m, _ = rollout(env, "rl", ctrls["rl"][1])
            comfort.append(m.comfort_wk_rms)
            viol.append(m.travel_violations)
            interv.append(m.safety_interventions)
        out["safety_on" if use_safety else "safety_off"] = {
            "comfort": float(np.mean(comfort)),
            "travel_violations": float(np.mean(viol)),
            "interventions": float(np.mean(interv)),
        }
    return out


def stage_fallback_table():
    sup = FallbackSupervisor(min_conf=0.4)
    rows = []
    cases = [
        ("healthy, good preview", 0.9, True, True, True),
        ("low preview confidence", 0.2, True, True, True),
        ("compute overrun", 0.9, True, False, True),
        ("estimator fault", 0.9, False, True, True),
        ("state diverging", 0.9, True, True, False),
    ]
    for name, conf, est_ok, comp_ok, state_ok in cases:
        mode = sup.decide(conf, est_ok, comp_ok, state_ok)
        rows.append((name, mode.value))
    return rows


# ---------------------------------------------------------------- plots
def plot_comfort_bar(nominal, out_path):
    scen = VAL_SCENARIOS
    ctrl_keys = [k for k in ("passive", "skyhook", "mpc", "rl", "bc")
                 if k in next(iter(nominal.values()))]
    x = np.arange(len(scen))
    w = 0.8 / len(ctrl_keys)
    plt.figure(figsize=(11, 5))
    for i, ck in enumerate(ctrl_keys):
        vals = [nominal[s][ck]["comfort_wk_rms"] for s in scen]
        plt.bar(x + i * w, vals, w, label=ck)
    plt.xticks(x + 0.4, scen, rotation=30, ha="right")
    plt.ylabel("ISO 2631 Wk-RMS vertical acc [m/s^2]")
    plt.title("Ride comfort by scenario (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def plot_sweep(d, out_path, xlabel, title):
    if not d:
        return
    xs = sorted(d.keys())
    ys = [d[x][0] for x in xs]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, "o-")
    plt.xlabel(xlabel)
    plt.ylabel("comfort Wk-RMS [m/s^2]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def plot_timeseries(ctrls, out_path):
    make = env_factory(randomize=False)
    plt.figure(figsize=(9, 4))
    for ck in ("passive", "mpc", "rl"):
        if ck not in ctrls:
            continue
        env = make(seed=42)
        env.set_scenario("single_bump")
        _, tr = rollout(env, ctrls[ck][0], ctrls[ck][1], record_trace=True)
        if tr is not None:
            plt.plot(tr.t, tr.heave_acc, label=ck)
    plt.xlabel("time [s]")
    plt.ylabel("body vertical acc [m/s^2]")
    plt.title("Single-bump response")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


# ---------------------------------------------------------------- report
def pct_improve(base, val):
    if base <= 1e-9:
        return 0.0
    return 100.0 * (base - val) / base


def write_report(out_dir, nominal, mc, prev_sweep, dr_sweep, safety, fb_rows):
    lines = ["# Active Suspension - Validation Report", ""]
    lines.append("## Stage 1: Nominal comfort (Wk-RMS [m/s^2], lower is better)")
    lines.append("")
    ctrl_keys = [k for k in ("passive", "skyhook", "mpc", "rl", "bc")
                 if k in next(iter(nominal.values()))]
    header = "| scenario | " + " | ".join(ctrl_keys) + " | RL vs passive |"
    lines.append(header)
    lines.append("|" + "---|" * (len(ctrl_keys) + 2))
    for s in VAL_SCENARIOS:
        row = [s]
        for ck in ctrl_keys:
            row.append(f"{nominal[s][ck]['comfort_wk_rms']:.3f}")
        if "rl" in ctrl_keys:
            imp = pct_improve(nominal[s]["passive"]["comfort_wk_rms"],
                              nominal[s]["rl"]["comfort_wk_rms"])
            row.append(f"{imp:+.1f}%")
        else:
            row.append("n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    # handling / constraints
    lines.append("## Stage 1b: Handling and constraints (nominal)")
    lines.append("")
    lines.append("| scenario | ctrl | sws_max [m] | dtl_ratio_max | travel_viol |")
    lines.append("|---|---|---|---|---|")
    for s in VAL_SCENARIOS:
        for ck in ctrl_keys:
            d = nominal[s][ck]
            lines.append(f"| {s} | {ck} | {d['sws_max']:.4f} | "
                         f"{d['dtl_ratio_max']:.3f} | {int(d['travel_violations'])} |")
    lines.append("")

    if mc:
        lines.append("## Stage 2: Monte Carlo (domain-randomized) comfort mean +/- std")
        lines.append("")
        mck = [k for k in ("passive", "mpc", "rl", "bc") if k in next(iter(mc.values()))]
        lines.append("| scenario | " + " | ".join(mck) + " |")
        lines.append("|" + "---|" * (len(mck) + 1))
        for s in VAL_SCENARIOS:
            row = [s]
            for ck in mck:
                d = mc[s][ck]
                row.append(f"{d['comfort_wk_rms']:.3f}+/-{d['comfort_wk_rms_std']:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if prev_sweep:
        lines.append("## Stage 3: Preview-error robustness (RL, iso_C)")
        lines.append("")
        lines.append("| preview-noise scale | comfort Wk-RMS |")
        lines.append("|---|---|")
        for k in sorted(prev_sweep):
            lines.append(f"| {k:.1f}x | {prev_sweep[k][0]:.3f}+/-{prev_sweep[k][1]:.3f} |")
        lines.append("")

    if dr_sweep:
        lines.append("## Stage 4: Domain-randomization width sweep (RL, iso_C)")
        lines.append("")
        lines.append("This addresses the open risk: too-narrow DR is fragile, "
                     "too-wide is over-conservative.")
        lines.append("")
        lines.append("| DR width | comfort Wk-RMS | mean travel violations |")
        lines.append("|---|---|---|")
        for k in sorted(dr_sweep):
            lines.append(f"| {k:.1f} | {dr_sweep[k][0]:.3f} | {dr_sweep[k][1]:.1f} |")
        lines.append("")

    if safety:
        lines.append("## Stage 5: Command-level safety filter (harsh speed_bump, DR width 1.5)")
        lines.append("")
        lines.append("| config | comfort | travel violations | interventions |")
        lines.append("|---|---|---|---|")
        for cfg_name, d in safety.items():
            lines.append(f"| {cfg_name} | {d['comfort']:.3f} | "
                         f"{d['travel_violations']:.1f} | {d['interventions']:.1f} |")
        lines.append("")

    lines.append("## Stage 6: Fallback supervisor decisions")
    lines.append("")
    lines.append("| health condition | mode |")
    lines.append("|---|---|")
    for name, mode in fb_rows:
        lines.append(f"| {name} | {mode} |")
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for fig in ["comfort_bar.png", "timeseries_single_bump.png",
                "preview_sweep.png", "dr_sweep.png"]:
        if os.path.exists(os.path.join(out_dir, fig)):
            lines.append(f"![{fig}]({fig})")
            lines.append("")

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", type=str, default="runs/ppo_main")
    ap.add_argument("--bc", type=str, default="runs/bc")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=str, default="runs/eval")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    actuator = default_config().actuator
    ctrls = load_controllers(args.rl, args.bc, actuator)
    print(f"[eval] controllers: {list(ctrls)}")

    print("[eval] stage 1: nominal report ...")
    nominal = stage_nominal(ctrls)
    print("[eval] stage 2: monte carlo ...")
    mc = stage_montecarlo(ctrls, args.seeds)
    print("[eval] stage 3: preview-error sweep ...")
    prev_sweep = stage_preview_sweep(ctrls, max(2, args.seeds // 2))
    print("[eval] stage 4: dr-width sweep ...")
    dr_sweep = stage_dr_sweep(ctrls, max(2, args.seeds // 2))
    print("[eval] stage 5: safety filter ...")
    safety = stage_safety(ctrls, max(2, args.seeds // 2))
    fb_rows = stage_fallback_table()

    print("[eval] plotting ...")
    plot_comfort_bar(nominal, os.path.join(args.out, "comfort_bar.png"))
    plot_timeseries(ctrls, os.path.join(args.out, "timeseries_single_bump.png"))
    plot_sweep(prev_sweep, os.path.join(args.out, "preview_sweep.png"),
               "preview-noise scale", "RL robustness to preview error")
    plot_sweep(dr_sweep, os.path.join(args.out, "dr_sweep.png"),
               "DR width", "RL comfort vs randomization width")

    all_metrics = {
        "nominal": nominal, "monte_carlo": mc,
        "preview_sweep": {str(k): v for k, v in prev_sweep.items()},
        "dr_sweep": {str(k): v for k, v in dr_sweep.items()},
        "safety": safety,
        "fallback": fb_rows,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    write_report(args.out, nominal, mc, prev_sweep, dr_sweep, safety, fb_rows)
    print(f"[done] report at {os.path.join(args.out, 'report.md')}")


if __name__ == "__main__":
    main()
