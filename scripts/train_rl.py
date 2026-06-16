"""Train the primary RL policy (raw 12-command action space).

Uses PPO (robust, on-policy) over domain-randomized episodes. Key sim-to-real
ingredients baked into the env: forward actuator with delay, command history in
the observation, command-rate/jerk penalty in the reward, and wide domain
randomization. Observation/reward normalization (VecNormalize) stabilizes
learning given the small road-height magnitudes.

Optionally warm-starts from MPC-teacher behavior-cloning weights produced by
collect_teacher.py (--warm_start path).

Usage:
    python scripts/train_rl.py --timesteps 300000 --n_envs 8 --out runs/ppo
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_suspension.config import default_config
from active_suspension.env import SuspensionEnv
from active_suspension.domain_randomization import DRConfig
from active_suspension.reward import RewardWeights


TRAIN_SCENARIOS = (
    "single_bump", "double_bump", "asymmetric_bump",
    "pothole", "speed_bump", "washboard", "iso_A", "iso_C", "iso_D",
)


def make_env_fn(seed: int, dr_width: float, use_safety: bool):
    def _fn():
        from stable_baselines3.common.monitor import Monitor
        cfg = default_config()
        env = SuspensionEnv(
            base_config=cfg,
            dr=DRConfig(enabled=dr_width > 0, width=dr_width),
            reward_weights=RewardWeights(),
            use_safety=use_safety,
            scenarios=TRAIN_SCENARIOS,
            randomize=dr_width > 0,
            seed=seed,
        )
        return Monitor(env)
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--dr_width", type=float, default=1.0)
    ap.add_argument("--use_safety", action="store_true")
    ap.add_argument("--out", type=str, default="runs/ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm_start", type=str, default="")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent_coef", type=float, default=0.0)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    venv = DummyVecEnv([
        make_env_fn(args.seed + i, args.dr_width, args.use_safety)
        for i in range(args.n_envs)
    ])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, gamma=0.99)

    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy", venv,
        learning_rate=args.lr,
        n_steps=1024,
        batch_size=4096,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
    )

    if args.warm_start:
        # warm_start is a directory containing a BC-distilled policy.zip
        # (+ optional vecnormalize.pkl) produced by collect_teacher.py.
        policy_zip = os.path.join(args.warm_start, "policy.zip")
        vn_pkl = os.path.join(args.warm_start, "vecnormalize.pkl")
        if os.path.exists(policy_zip):
            try:
                model.set_parameters(os.path.join(args.warm_start, "policy"),
                                     exact_match=False)
                if os.path.exists(vn_pkl):
                    import pickle
                    with open(vn_pkl, "rb") as f:
                        vn = pickle.load(f)
                    venv.obs_rms = vn.obs_rms  # reuse teacher obs statistics
                print(f"[warm-start] loaded teacher-distilled policy from {args.warm_start}")
            except Exception as e:
                print(f"[warm-start] failed ({e}); training from scratch")
        else:
            print(f"[warm-start] no policy at {policy_zip}; training from scratch")

    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    model.save(os.path.join(args.out, "policy"))
    venv.save(os.path.join(args.out, "vecnormalize.pkl"))
    print(f"[done] saved policy + vecnormalize to {args.out}")


if __name__ == "__main__":
    main()
