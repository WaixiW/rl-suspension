"""Deterministic, stratified, scenario-safe split generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np

from rl_suspension.production.contracts import Scenario
from rl_suspension.production.provenance import canonical_json_bytes, canonical_sha256


SUPPORTED_SPLITS = ("train", "validation", "test")
SUPPORTED_FAMILIES = ("flat", "single_bump", "double_bump", "asymmetric_bump")


@dataclass(frozen=True)
class ScenarioSpace:
    """Continuous scenario ranges sampled with per-family Latin hypercubes."""

    families: tuple[str, ...] = SUPPORTED_FAMILIES
    speed_mps: tuple[float, float] = (5.0, 30.0)
    bump_height_m: tuple[float, float] = (0.015, 0.09)
    bump_width_m: tuple[float, float] = (0.25, 1.2)
    asymmetry: tuple[float, float] = (0.0, 0.8)
    double_spacing_m: tuple[float, float] = (0.4, 2.0)
    bump_start_m: tuple[float, float] = (1.5, 3.5)
    episode_steps: int = 250

    def validate(self) -> None:
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("families must be nonempty and unique")
        unknown = set(self.families).difference(SUPPORTED_FAMILIES)
        if unknown:
            raise ValueError(f"unsupported bump families: {sorted(unknown)}")
        for name in (
            "speed_mps",
            "bump_height_m",
            "bump_width_m",
            "asymmetry",
            "double_spacing_m",
            "bump_start_m",
        ):
            low, high = getattr(self, name)
            if not np.isfinite(low) or not np.isfinite(high) or low > high:
                raise ValueError(f"invalid range for {name}")
        if self.speed_mps[0] <= 0.0 or self.episode_steps <= 0:
            raise ValueError("speed and episode_steps must be positive")


class ScenarioGenerator:
    """Generate exact split counts while balancing every split across families."""

    def __init__(self, seed: int = 0, space: ScenarioSpace | None = None) -> None:
        self.seed = int(seed)
        self.space = space or ScenarioSpace()
        self.space.validate()

    def generate(self, split_counts: Mapping[str, int]) -> list[Scenario]:
        unknown = set(split_counts).difference(SUPPORTED_SPLITS)
        if unknown:
            raise ValueError(f"unsupported scenario splits: {sorted(unknown)}")
        if any(int(count) != count or count < 0 for count in split_counts.values()):
            raise ValueError("split counts must be nonnegative integers")

        scenarios: list[Scenario] = []
        for split in SUPPORTED_SPLITS:
            count = int(split_counts.get(split, 0))
            scenarios.extend(self._generate_split(split, count))
        validate_split_safety(scenarios)
        return scenarios

    def generate_split(self, split: str, count: int) -> list[Scenario]:
        return self.generate({split: count})

    def _generate_split(self, split: str, count: int) -> list[Scenario]:
        if count == 0:
            return []
        family_counts = _balanced_counts(count, len(self.space.families))
        scenarios: list[Scenario] = []
        ordinal = 0
        for family, family_count in zip(self.space.families, family_counts):
            if family_count == 0:
                continue
            unit_samples = self._latin_hypercube(
                split=split,
                family=family,
                count=family_count,
                dimensions=6,
            )
            for family_ordinal, unit in enumerate(unit_samples):
                parameters = self._parameters(family, unit)
                scenario_seed = _derived_seed(
                    self.seed,
                    split,
                    family,
                    family_ordinal,
                )
                identity = {
                    "generator_seed": self.seed,
                    "split": split,
                    "family": family,
                    "family_ordinal": family_ordinal,
                    "scenario_seed": scenario_seed,
                    "parameters": parameters,
                    "space": asdict(self.space),
                }
                scenario_id = f"{split}-{family}-{canonical_sha256(identity)[:16]}"
                scenarios.append(
                    Scenario(
                        scenario_id=scenario_id,
                        seed=scenario_seed,
                        split=split,
                        bump_family=family,
                        parameters=parameters,
                    )
                )
                ordinal += 1
        if ordinal != count:
            raise RuntimeError("internal stratification count mismatch")
        return scenarios

    def _latin_hypercube(
        self,
        *,
        split: str,
        family: str,
        count: int,
        dimensions: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(_derived_seed(self.seed, split, family, "lhs"))
        values = np.empty((count, dimensions), dtype=np.float64)
        for dimension in range(dimensions):
            permutation = rng.permutation(count)
            values[:, dimension] = (permutation + rng.random(count)) / count
        return values

    def _parameters(self, family: str, unit: np.ndarray) -> dict[str, float | int]:
        interpolate = lambda bounds, value: float(  # noqa: E731
            bounds[0] + value * (bounds[1] - bounds[0])
        )
        parameters: dict[str, float | int] = {
            "speed_mps": interpolate(self.space.speed_mps, unit[0]),
            "bump_start_m": interpolate(self.space.bump_start_m, unit[4]),
            "episode_steps": self.space.episode_steps,
        }
        if family == "flat":
            parameters.update(
                bump_height_m=0.0,
                bump_width_m=interpolate(self.space.bump_width_m, unit[2]),
                asymmetry=0.0,
                double_spacing_m=interpolate(self.space.double_spacing_m, unit[5]),
            )
        else:
            parameters.update(
                bump_height_m=interpolate(self.space.bump_height_m, unit[1]),
                bump_width_m=interpolate(self.space.bump_width_m, unit[2]),
                asymmetry=interpolate(self.space.asymmetry, unit[3]),
                double_spacing_m=interpolate(self.space.double_spacing_m, unit[5]),
            )
        return parameters


def generate_stratified_scenarios(
    split_counts: Mapping[str, int],
    *,
    seed: int = 0,
    space: ScenarioSpace | None = None,
) -> list[Scenario]:
    return ScenarioGenerator(seed=seed, space=space).generate(split_counts)


def scenario_fingerprint(scenario: Scenario, *, include_split: bool = False) -> str:
    """Hash physical identity; excluding split is useful for leakage checks."""

    payload = {
        "bump_family": scenario.bump_family,
        "parameters": scenario.parameters,
        "version": scenario.version,
    }
    if include_split:
        payload["split"] = scenario.split
    return canonical_sha256(payload)


def validate_split_safety(scenarios: Iterable[Scenario]) -> None:
    """Reject duplicate IDs or physical scenario identities across splits."""

    ids: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    seeds: dict[int, str] = {}
    for scenario in scenarios:
        if scenario.split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split {scenario.split!r}")
        if scenario.scenario_id in ids:
            prior_split = ids[scenario.scenario_id]
            if prior_split != scenario.split:
                raise ValueError(
                    f"scenario_id {scenario.scenario_id!r} leaks across splits"
                )
            raise ValueError(f"duplicate scenario_id {scenario.scenario_id!r}")
        ids[scenario.scenario_id] = scenario.split

        fingerprint = scenario_fingerprint(scenario)
        prior_split = fingerprints.setdefault(fingerprint, scenario.split)
        if prior_split != scenario.split:
            raise ValueError("physical scenario identity leaks across splits")

        prior_split = seeds.setdefault(int(scenario.seed), scenario.split)
        if prior_split != scenario.split:
            raise ValueError(f"scenario seed {scenario.seed} leaks across splits")


def scenario_payloads(scenarios: Sequence[Scenario]) -> list[dict[str, object]]:
    return [asdict(scenario) for scenario in scenarios]


def _balanced_counts(total: int, groups: int) -> list[int]:
    base, remainder = divmod(total, groups)
    return [base + int(index < remainder) for index in range(groups)]


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256(canonical_json_bytes(parts)).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)
