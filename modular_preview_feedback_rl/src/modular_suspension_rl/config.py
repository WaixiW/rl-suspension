from dataclasses import dataclass, field
from typing import Tuple


WHEEL_ORDER: Tuple[str, ...] = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class ObservationConfig:
    preview_range_m: float = 8.0
    preview_resolution_m: float = 0.05
    wheelbase_m: float = 2.8
    min_speed_mps: float = 0.5
    preview_feature_count: int = 6
    feedback_dim: int = 64
    feedback_history: int = 8
    normalization_epsilon: float = 1e-6

    @property
    def preview_points(self) -> int:
        return int(round(self.preview_range_m / self.preview_resolution_m))

    def validate(self) -> None:
        if self.preview_points != 160:
            raise ValueError("The default ADS contract requires exactly 160 preview points.")
        if self.feedback_dim <= 0 or self.feedback_history <= 0:
            raise ValueError("Feedback dimensions must be positive.")


@dataclass(frozen=True)
class NetworkConfig:
    preview_channels: Tuple[int, ...] = (64, 96, 128)
    preview_kernel_size: int = 5
    state_hidden_dim: int = 128
    state_layers: int = 1
    fused_hidden_dim: int = 192
    force_limit_n: float = 5000.0
    preview_residual_limit_n: float = 2500.0
    gate_smoothing: float = 0.85


@dataclass(frozen=True)
class ActuatorConfig:
    sample_time_s: float = 0.002
    force_time_constant_s: float = 0.025
    passive_damping_ns_per_m: float = 1200.0
    current_damping_gain_ns_per_m_per_a: float = 850.0
    pump_force_gain_n_per_rpm: float = 1.5
    direction_softness: float = 25.0
    current_bounds_a: Tuple[float, float] = (0.0, 3.0)
    rpm_bounds: Tuple[float, float] = (-4000.0, 4000.0)
    current_slew_a_per_s: float = 80.0
    rpm_slew_per_s: float = 60000.0
    force_bounds_n: Tuple[float, float] = (-6000.0, 6000.0)
    allocator_iterations: int = 28
    allocator_step_size: float = 0.12
    energy_weight: float = 1e-7
    slew_weight: float = 2e-4


@dataclass(frozen=True)
class RewardScales:
    vertical_acceleration_mps2: float = 5.0
    angular_rate_radps: float = 0.5
    angular_acceleration_radps2: float = 3.0
    jerk_mps3: float = 20.0
    suspension_travel_m: float = 0.08
    tire_load_n: float = 3500.0
    wheel_hop_mps: float = 0.5
    electrical_power_w: float = 4000.0
    command_slew: float = 1.0
    force_error_n: float = 500.0


@dataclass(frozen=True)
class RewardWeights:
    vertical_acceleration: float = 1.0
    pitch: float = 0.25
    roll: float = 0.35
    passenger_acceleration: float = 0.5
    jerk: float = 0.05
    suspension_travel: float = 0.4
    tire_load_variation: float = 0.35
    wheel_hop: float = 0.15
    electrical_power: float = 0.02
    command_slew: float = 0.02
    force_tracking: float = 0.1
    gate_smoothness: float = 0.01
    invalid_preview_use: float = 0.1
    hard_violation: float = 100.0


@dataclass(frozen=True)
class RewardConfig:
    scales: RewardScales = field(default_factory=RewardScales)
    weights: RewardWeights = field(default_factory=RewardWeights)
    travel_soft_limit_m: float = 0.07
    tire_load_min_n: float = 100.0


@dataclass(frozen=True)
class TD3Config:
    discount: float = 0.99
    target_tau: float = 0.005
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    policy_delay: int = 2
    target_noise_std: float = 0.12
    target_noise_clip: float = 0.3
    exploration_noise_std: float = 0.1
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    gate_regularization_weight: float = 0.02
    projection_penalty_weight: float = 0.02
