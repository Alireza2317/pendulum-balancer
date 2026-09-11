from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
	# Buffer and training
	buffer_maxsize: int = 100_000
	buffer_warmup_size: int = 4_000
	max_episodes: int = 30
	batch_size: int = 128

	# Agent hyperparameters
	## Learning rates
	actor_lr: float = 2e-7
	critic_lr: float = 1e-4

	## Discount factor
	gamma: float = 0.998

	## Polyak averaging coefficient
	tau: float = 0.004

	## Policy parameters
	policy_delay: int = 2  # actor + targets update every N critic updates
	policy_noise: float = 0.2  # target smoothing noise stddev (in [-1,1] action space)
	noise_clip: float = 0.5  # clip range for that smoothing noise

	# Physics and environment
	max_force: float = 60.0

	# Time step
	dt: float = 1.0 / 240.0

	## State and action
	## 8 = cart_x, cart_x_vel, sin(a1), cos(a1), a1_vel, sin(a2), cos(a2), a2_vel
	state_dim: int = 8
	action_dim: int = 1

	## Gravity
	gravity: float = -9.81

	## Damping (simulating friction)
	joint_damping: float = 0.005

	## Max poles velocities
	max_velocity: float = 10

	## Absolute path of the pendulum urdf file
	pendulum_urdf_path: str = str(PROJECT_ROOT / "assets" / "urdf" / "pendulum.urdf")

	## Rewards
	terminal_penalty: float = 1.0
	### A positive constant to encourage staying alive
	alive_bonus: float = 6.5

	### Positive progess reward, for immediate local payoff for angle recovery
	# progress_reward_scale: float = 2.0
	progress_reward_scale: float = 0
	### Clamp on raw (pre-scale) cost delta, in case of pathological single-step swing
	progress_cost_clip: float = 1.0

	## Episode termination limitations
	cart_x_threshold: float = 0.95
	## Maximum episode length, independent of angle. Needed because once the
	## curriculum avoids angle-based termination, angle alone won't end episodes.
	max_episode_steps: int = 1200  # Equivalent to 5s at 240Hz

	# Curriculum
	## How far (deg) from vertical each pole is randomized at reset.
	curriculum_reset_start_deg: float = 5.0
	curriculum_reset_end_deg: float = 180.0

	## Angle threshold is the randomized angle + the margin
	## At curriculum_level=0: near-vertical reset, tight threshold (pure balance).
	## At curriculum_level=1: reset from fully hanging (180 deg), threshold effectively
	## disabled (>180 deg, i.e. angle can never trigger it)
	curriculum_margin_start_deg: float = 12.0
	curriculum_margin_end_deg: float = 181.0

	## Rolling window (in episodes) used to judge whether the agent has mastered
	## the current difficulty level.
	curriculum_window: int = 40

	## Fraction of episodes in the rolling window that must complete successfully
	## before the curriculum advances.
	curriculum_success_ratio: float = 0.7
	curriculum_success_ratio_min: float = 0.4

	curriculum_episode_balance_threshold: float = 0.7

	## How much curriculum_level (0..1) increases each time the success bar is met.
	curriculum_step: float = 0.005

	## Level advancement requirements
	advancement_base_successes: int = 5
	advancement_current_successes: int = 4
	advancement_current_survival_rate: float = 0.9

	## Regression stopping
	regression_base_min_successes: int = 4
	regression_base_min_survival: float = 0.95
	regression_patience: int = 2

	# Balance metrics
	balance_angle_threshold_deg: float = 5.0
	balance_velocity_threshold: float = 1.0
	balance_cart_threshold: float = 0.5

	# Exploration
	## OU noise parameters
	ounoise_mu: float = 0.0
	ounoise_theta: float = 0.15
	ounoise_sigma: float = 0.20
	ounoise_sigma_min: float = 0.05
	ounoise_sigma_min_max: float = 0.15
	ounoise_decay: float = 0.9993

	stagnation_reheat_interval: int = 300

	## If success ratio is bigger than this, noise decays
	exploration_decay_unlock_threshold: float = 0.1

	## Number of environment steps taken before triggering a network update
	## It means the agent acts n times in the simulation per 1 training step.
	train_every_n_steps: int = 4

	# Reset schedule
	targeted_reset_cycle: int = 4
	targeted_reset_count: int = 2

	# Checkpointing frequency
	checkpoint_every_n_episodes: int = 5

	# Evaluation
	evaluate_every_n_episodes: int = 5
	evaluate_gate_retry_episodes: int = 10

	# Reproducibility
	seed: int = 23

	def __post_init__(self) -> None:
		if self.buffer_warmup_size < self.batch_size:
			raise ValueError("Buffer warmup size should be >= batch size!")
		if self.buffer_maxsize < self.buffer_warmup_size:
			raise ValueError("Buffer maxsize should be >= warmup size!")
		if self.train_every_n_steps <= 0:
			raise ValueError("train_every_n_steps must be a positive integer!")
		if self.checkpoint_every_n_episodes <= 0:
			raise ValueError("checkpoint_every_n_episodes must be a positive integer!")
		if self.policy_delay <= 0:
			raise ValueError("policy_delay must be a positive integer!")
		if self.batch_size <= 0:
			raise ValueError("batch_size must be a positive integer!")
		if self.max_episode_steps <= 0:
			raise ValueError("max_episode_steps must be a positive integer!")
		if self.curriculum_window <= 0:
			raise ValueError("curriculum_window must be a positive integer!")
		if self.dt <= 0:
			raise ValueError("dt (Physics time step) must be positive!")
		if self.balance_angle_threshold_deg <= 0:
			raise ValueError("balance_angle_threshold must be positive!")
		if self.balance_velocity_threshold <= 0:
			raise ValueError("balance_velocity_threshold must be positive!")
		if self.balance_cart_threshold <= 0:
			raise ValueError("balance_cart_threshold must be positive!")
		if not 0.0 <= self.curriculum_episode_balance_threshold <= 1.0:
			raise ValueError("curriculum_episode_balance_threshold must be in [0, 1].")
		if self.evaluate_every_n_episodes <= 0:
			raise ValueError("evaluation_every_n_episodes must be positive!")
		if self.evaluate_gate_retry_episodes <= 0:
			raise ValueError("evaluation_gate_retry_episodes must be positive!")
		if not 0 <= self.targeted_reset_count <= self.targeted_reset_cycle:
			raise ValueError(
				"targeted_reset_count must be between 0 and targeted_reset_cycle"
			)
		if self.targeted_reset_cycle <= 0:
			raise ValueError("targeted_reset_cycle must be positive!")

		if (
			self.targeted_reset_count > 0
			and self.targeted_reset_cycle % self.targeted_reset_count != 0
		):
			raise ValueError(
				"targeted_reset_cycle must be divisible by targeted_reset_count"
			)

		if self.regression_patience <= 0:
			raise ValueError("regression_patience must be positive!")

	def save(self, filepath: Path | str) -> None:
		filepath = Path(filepath)
		filepath.write_text(json.dumps(asdict(self), indent=4))

	@classmethod
	def load(cls, filepath: Path | str) -> Config:
		filepath = Path(filepath)
		return cls(**json.loads(filepath.read_text()))
