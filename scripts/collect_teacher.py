"""MPC teacher -> command labels -> behavior-cloned policy (offline).

This realizes the "MPC as offline teacher" role:
  1. Run the preview-LQR MPC teacher across scenarios. The teacher outputs
     desired forces; the offline actuator inverse (Path B) converts them to
     command labels. NONE of this is on the runtime path.
  2. Collect (observation, command-label) pairs.
  3. Behavior-clone a PPO-compatible policy on the pairs and save it together
     with VecNormalize statistics, so it can be evaluated directly or used to
     warm-start RL (train_rl.py --warm_start runs/bc/policy.zip).

Usage:
    python scripts/collect_teacher.py --episodes 40 --out runs/bc
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_suspension.config import default_config
from active_suspension.env import SuspensionEnv
from active_suspension.mpc import MPCTeacher
from active_suspension.actuator import ForwardActuator
from active_suspension.domain_randomization import DRConfig


SCENARIOS = (
    "single_bump", "double_bump", "asymmetric_bump",
    "pothole", "speed_bump", "washboard", "iso_A", "iso_C", "iso_D",
)


def collect(episodes: int, dr_width: float, seed: int):
    cfg = default_config()
    obs_buf, act_buf = [], []
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        scen = SCENARIOS[ep % len(SCENARIOS)]
        env = SuspensionEnv(
            cfg, dr=DRConfig(enabled=dr_width > 0, width=dr_width),
            randomize=dr_width > 0, scenarios=(scen,),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        env.set_scenario(scen)
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        teacher = MPCTeacher(env.car, env.cfg.actuator, env.cfg.sim.control_dt)
        for _ in range(env._max_control_steps()):
            inp = env.controller_inputs()
            cmd, _ = teacher.act(inp["x_hat"], inp["rel_vel"], inp["road_future"])
            label = ForwardActuator.denormalize_command(cmd, env.cfg.actuator)
            obs_buf.append(np.asarray(obs, np.float32))
            act_buf.append(np.asarray(label, np.float32))
            res = env.apply_command(cmd)
            obs = res.obs
            if res.terminated:
                break
    return np.array(obs_buf), np.array(act_buf)


def behavior_clone(obs, acts, out_dir, epochs=40, batch=2048, lr=1e-3, seed=0):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    os.makedirs(out_dir, exist_ok=True)
    cfg = default_config()

    # VecNormalize with obs stats taken from the teacher data.
    venv = DummyVecEnv([lambda: SuspensionEnv(cfg, randomize=False)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    venv.obs_rms.mean = obs.mean(axis=0)
    venv.obs_rms.var = obs.var(axis=0) + 1e-8
    venv.obs_rms.count = obs.shape[0]
    venv.training = False

    model = PPO("MlpPolicy", venv, policy_kwargs=dict(net_arch=[256, 256]),
                device="cpu", seed=seed)

    norm = np.clip((obs - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var + 1e-8),
                   -10.0, 10.0).astype(np.float32)
    X = torch.as_tensor(norm)
    Y = torch.as_tensor(acts)
    n = X.shape[0]
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        idx = rng.permutation(n)
        losses = []
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            xb, yb = X[b], Y[b]
            dist = model.policy.get_distribution(xb)
            mean = dist.distribution.mean
            loss = ((mean - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  BC epoch {ep+1:3d}/{epochs}  mse={np.mean(losses):.4f}")

    model.save(os.path.join(out_dir, "policy"))
    venv.save(os.path.join(out_dir, "vecnormalize.pkl"))
    print(f"[done] saved BC policy + vecnormalize to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--dr_width", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", type=str, default="runs/bc")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[collect] {args.episodes} teacher episodes ...")
    obs, acts = collect(args.episodes, args.dr_width, args.seed)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "teacher_data.npz"),
                        obs=obs, acts=acts)
    print(f"[collect] dataset: obs={obs.shape}, acts={acts.shape}")
    print(f"[bc] behavior cloning ...")
    behavior_clone(obs, acts, args.out, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
