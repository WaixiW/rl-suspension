"""Production evaluation, statistics, and promotion gates."""

from rl_suspension.production.evaluation.closed_loop import (
    ClosedLoopEpisode,
    ControllerSummary,
    PairedClosedLoopEvaluator,
    PairedClosedLoopReport,
    evaluate_paired_closed_loop,
)
from rl_suspension.production.evaluation.gates import (
    GateCheck,
    PromotionCriteria,
    PromotionDecision,
    PromotionEvidence,
    PromotionGateConfig,
    evaluate_promotion_gates,
)
from rl_suspension.production.evaluation.metrics import (
    BootstrapInterval,
    ChannelErrorMetrics,
    OpenLoopReport,
    bootstrap_confidence_interval,
    open_loop_metrics,
)

__all__ = [
    "BootstrapInterval",
    "ChannelErrorMetrics",
    "ClosedLoopEpisode",
    "ControllerSummary",
    "GateCheck",
    "OpenLoopReport",
    "PairedClosedLoopEvaluator",
    "PairedClosedLoopReport",
    "PromotionCriteria",
    "PromotionDecision",
    "PromotionEvidence",
    "PromotionGateConfig",
    "bootstrap_confidence_interval",
    "evaluate_paired_closed_loop",
    "evaluate_promotion_gates",
    "open_loop_metrics",
]
