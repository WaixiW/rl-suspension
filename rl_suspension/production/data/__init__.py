"""Production direct-12D collection, validation, and dataset utilities."""

from rl_suspension.production.data.card import (
    build_data_card,
    generate_data_card,
    write_data_card,
)
from rl_suspension.production.data.collection import (
    CollectionSummary,
    EpisodePhase,
    PHASE_NAMES,
    collect_episode,
    collect_scenarios,
    infer_episode_phase,
)
from rl_suspension.production.data.dataset import (
    DATASET_VERSION,
    Direct12Dataset,
    EpisodeRecord,
    EpisodeShardWriter,
    iter_episode_shards,
    load_dataset,
    validate_episode_arrays,
)
from rl_suspension.production.data.diagnostics import (
    ActionAmbiguityReport,
    AmbiguousPair,
    action_ambiguity_diagnostic,
    diagnose_action_ambiguity,
)
from rl_suspension.production.data.normalization import (
    GroupedNormalization,
    NormalizationGroup,
    compute_grouped_normalization,
    default_group_specs,
)
from rl_suspension.production.data.sampling import PhaseBalancedSampler
from rl_suspension.production.data.training_bridge import to_bc_dataset
from rl_suspension.production.data.validation import (
    DatasetValidationReport,
    ValidationIssue,
    validate_dataset,
)

__all__ = [
    "DATASET_VERSION",
    "ActionAmbiguityReport",
    "AmbiguousPair",
    "CollectionSummary",
    "DatasetValidationReport",
    "Direct12Dataset",
    "EpisodePhase",
    "EpisodeRecord",
    "EpisodeShardWriter",
    "GroupedNormalization",
    "NormalizationGroup",
    "PHASE_NAMES",
    "PhaseBalancedSampler",
    "ValidationIssue",
    "action_ambiguity_diagnostic",
    "build_data_card",
    "collect_episode",
    "collect_scenarios",
    "compute_grouped_normalization",
    "default_group_specs",
    "diagnose_action_ambiguity",
    "generate_data_card",
    "infer_episode_phase",
    "iter_episode_shards",
    "load_dataset",
    "to_bc_dataset",
    "validate_dataset",
    "validate_episode_arrays",
    "write_data_card",
]
