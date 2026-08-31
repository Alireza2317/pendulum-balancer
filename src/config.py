from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
	# Buffer and training
	buffer_maxsize: int = 300_000
	buffer_warmup_size: int = 3_000
	max_episodes: int = 3_000
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
	max_force: float = 15.0

	## Gravity
	gravity: float = -9.81

	## Damping (simulating friction)
	joint_damping: float = 0.005

	## Max poles velocities
	max_velocity: float = 10

	## Maximum random deviation (degrees) from vertical when resetting the environment.
	reset_angle_range_deg: float = 40.0

	## Absolute path of the pendulum urdf file
	pendulum_urdf_path: str = "assets/urdf/pendulum.urdf"

	## Rewards
	terminal_penalty: float = 1.0
	### A small positive constant to encourage staying alive
	alive_bonus: float = 5e-2

	## Episode termination limitations
	cart_x_threshold: float = 0.95
	angle_threshold_deg: float = 15.0

	# Exploration
	action_noise_std: float = 0.1
	noise_decay: float = 0.995
	min_action_noise_std: float = 1e-2

	## Number of environment steps taken before triggering a network update
	## It means the agent acts n times in the simulation per 1 training step.
	train_every_n_steps: int = 4

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
