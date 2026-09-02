from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
	# Buffer and training
	buffer_maxsize: int = 300_000
	buffer_warmup_size: int = 3_000
	max_episodes: int = 12_000
	batch_size: int = 128

	# Agent hyperparameters
	## Learning rates
	actor_lr: float = 1e-4
	critic_lr: float = 1e-3

	## Discount factor
	gamma: float = 0.99

	## Polyak averaging coefficient
	tau: float = 0.001

	# Physics and environment
	max_force: float = 30.0

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
	pendulum_urdf_path: str = "assets/urdf/pendulum.urdf"

	## Rewards
	terminal_penalty: float = 1.0
	### A small positive constant to encourage staying alive
	alive_bonus: float = 5e-2

	## Episode termination limitations
	cart_x_threshold: float = 0.95
	## Maximum episode length, independent of angle. Needed because once the
	## curriculum avoids angle-based termination, angle alone won't end episodes.
	max_episode_steps: int = 500

	# Curriculum
	## How far (deg) from vertical each pole is randomized at reset.
	curriculum_reset_start_deg: float = 5.0
	curriculum_reset_end_deg: float = 180.0

	## Angle threshold is the randomized angle + the margin
	## At curriculum_level=0: near-vertical reset, tight threshold (pure balance).
	## At curriculum_level=1: reset from fully hanging (180 deg), threshold effectively
	## disabled (>180 deg, i.e. angle can never trigger it)
	curriculum_margin_start_deg: float = 10.0
	curriculum_margin_end_deg: float = 181.0

	## Rolling window (in episodes) used to judge whether the agent has mastered
	## the current difficulty level.
	curriculum_window: int = 50

	## Fraction of max_episode_steps the agent must survive on average, over the
	## window, before curriculum difficulty is increased.
	curriculum_success_ratio: float = 0.9

	## How much curriculum_level (0..1) increases each time the success bar is met.
	curriculum_step: float = 0.05

	# Exploration
	## OU noise parameters
	ounoise_mu: float = 0.0
	ounoise_theta: float = 0.15
	ounoise_sigma: float = 0.2
	ounoise_sigma_min: float = 0.05
	ounoise_decay: float = 0.9997

	## If success ratio is bigger than this, noise decays
	exploration_decay_unlock_threshold: float = 0.25

	## Number of environment steps taken before triggering a network update
	## It means the agent acts n times in the simulation per 1 training step.
	train_every_n_steps: int = 2

	# Logging and checkpointing
	log_every_n_episodes: int = 20

	def __post_init__(self) -> None:
		if self.buffer_warmup_size < self.batch_size:
			raise ValueError("Buffer warmup size should be bigger than batch size!")

	def save(self, filepath: Path | str) -> None:
		filepath = Path(filepath)
		filepath.write_text(json.dumps(asdict(self), indent=4))

	@classmethod
	def load(cls, filepath: Path | str) -> Config:
		filepath = Path(filepath)
		return cls(**json.loads(filepath.read_text()))
