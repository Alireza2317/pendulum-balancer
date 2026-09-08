from pathlib import Path

import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.trainer.trainer import DDPGTrainer


class ModelCheckpointer:
	def __init__(
		self,
		agent: DDPGAgent,
		trainer: DDPGTrainer | None = None,
		log_dir: Path | str = Path("logs"),
		checkpoint_dir: Path | str = Path("checkpoints"),
		max_to_keep: int = 10,
	) -> None:
		self.log_dir = Path(log_dir)
		self.checkpoint_dir = Path(checkpoint_dir)
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
		self.trainer = trainer
		self.episode_counter = tf.Variable(
			tf.constant(0, dtype=tf.uint64), trainable=False
		)

		# Persists CurriculumManager.level across process restarts.
		self.curriculum_level = tf.Variable(
			tf.constant(0.0, dtype=tf.float32), trainable=False
		)

		# OU noise sigma since it changes over time

		self.ounoise_sigma = tf.Variable(
			tf.constant(
				0 if trainer is None else trainer.noise.sigma, dtype=tf.float32
			),
			trainable=False,
		)

		self.writer = tf.summary.create_file_writer(str(self.log_dir))
		self.checkpoint = tf.train.Checkpoint(
			actor=agent.actor,
			critic=agent.critic,
			critic2=agent.critic2,
			target_actor=agent.target_actor,
			target_critic=agent.target_critic,
			target_critic2=agent.target_critic2,
			actor_optimizer=agent.actor_optimizer,
			critic_optimizer=agent.critic_optimizer,
			episode=self.episode_counter,
			curriculum_level=self.curriculum_level,
			noise_sigma=self.ounoise_sigma,
		)
		self.manager = tf.train.CheckpointManager(
			self.checkpoint, directory=str(self.checkpoint_dir), max_to_keep=max_to_keep
		)

	def save(self, step: int) -> str | None:
		self.episode_counter.assign(step)
		if self.trainer is not None:
			self.curriculum_level.assign(self.trainer.curriculum.level)
			self.ounoise_sigma.assign(self.trainer.noise.sigma)

			# Delete all old window files
			for old_window in self.checkpoint_dir.glob("window_*.pkl"):
				old_window.unlink(missing_ok=True)
			# Save the window
			window_path = self.checkpoint_dir / f"window_{step}.pkl"
			self.trainer.curriculum.save(window_path)

			# Delete all old buffer files
			for old_window in self.checkpoint_dir.glob("buffer_*.pkl"):
				old_window.unlink(missing_ok=True)
			# Save the buffer
			buffer_path = self.checkpoint_dir / f"buffer_{step}.pkl"
			self.trainer.buffer.save(buffer_path)

		return self.manager.save(checkpoint_number=step)

	def load_latest(self) -> bool:
		if self.manager.latest_checkpoint:
			self.checkpoint.restore(self.manager.latest_checkpoint)
			if self.trainer is not None:
				self.trainer.curriculum.set_level(float(self.curriculum_level))
				self.trainer.noise.set_sigma(float(self.ounoise_sigma))

				episode: int = int(self.episode_counter)
				if episode <= 0:
					return True

				# Load window
				window_path = self.checkpoint_dir / f"window_{episode}.pkl"
				if window_path.exists():
					self.trainer.curriculum.load(window_path)

				# Load buffer
				buffer_path = self.checkpoint_dir / f"buffer_{episode}.pkl"
				if buffer_path.exists():
					self.trainer.buffer.load(buffer_path)

			return True

		return False

	def log_scalar(self, name: str, value: float, step: int) -> None:
		with self.writer.as_default():
			tf.summary.scalar(name, value, step=step)
