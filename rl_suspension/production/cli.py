"""Private-server production pipeline command line."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rl_suspension.production.adapters import load_plugin
from rl_suspension.production.certification import certify_integration
from rl_suspension.production.contracts import Scenario
from rl_suspension.production.reference import (
    ReferenceDirect12Simulator,
    ReferenceMpcAdapter,
)


def _scenario(path: Path | None) -> Scenario:
    if path is None:
        return Scenario(
            scenario_id="contract-reference",
            seed=0,
            split="validation",
            bump_family="single_bump",
            parameters={"episode_steps": 10},
        )
    return Scenario(**json.loads(path.read_text(encoding="utf-8")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    certify = subparsers.add_parser("certify", help="certify MPC/simulator contracts")
    certify.add_argument(
        "--mpc-plugin",
        default=os.environ.get("MPC_PLUGIN"),
        help="package.module:factory; omit with --reference",
    )
    certify.add_argument(
        "--simulator-plugin",
        default=os.environ.get("SIMULATOR_PLUGIN"),
    )
    certify.add_argument("--scenario-json", type=Path, default=None)
    certify.add_argument("--output", type=Path, required=True)
    certify.add_argument("--reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "certify":
        if args.reference:
            mpc, simulator = ReferenceMpcAdapter(), ReferenceDirect12Simulator()
        else:
            if not args.mpc_plugin or not args.simulator_plugin:
                raise ValueError("MPC and simulator plugin specifications are required")
            mpc = load_plugin(args.mpc_plugin)
            simulator = load_plugin(args.simulator_plugin)
        report = certify_integration(mpc, simulator, _scenario(args.scenario_json))
        report.save(args.output)
        print(json.dumps(report.__dict__, indent=2))
        if not report.passed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
