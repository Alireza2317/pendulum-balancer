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
		self.trainer = trainer
		self.episode_counter = tf.Variable(0, dtype=tf.uint64)

		# Persists CurriculumManager.level across process restarts.
		self.curriculum_level = tf.Variable(0.0, dtype=tf.float32)

		# OU noise sigma since it changes over time

		self.ounoise_sigma = tf.Variable(
			0 if trainer is None else trainer.noise.sigma, dtype=tf.float32
		)

		self.writer = tf.summary.create_file_writer(str(self.log_dir))
		self.checkpoint = tf.train.Checkpoint(
			actor=agent.actor,
			critic=agent.critic,
			target_actor=agent.target_actor,
			target_critic=agent.target_critic,
			episode=self.episode_counter,
			curriculum_level=self.curriculum_level,
			noise_sigma=self.ounoise_sigma,
		)
		self.manager = tf.train.CheckpointManager(
			self.checkpoint, directory=str(self.checkpoint_dir), max_to_keep=max_to_keep
		)

	def save(self, step: int) -> str | None:
		if self.trainer is not None:
			self.curriculum_level.assign(self.trainer.curriculum.level)
			self.ounoise_sigma.assign(self.trainer.noise._sigma)

		return self.manager.save(checkpoint_number=step)

	def load_latest(self) -> bool:
		if self.manager.latest_checkpoint:
			self.checkpoint.restore(self.manager.latest_checkpoint)
			if self.trainer is not None:
				self.trainer.curriculum.set_level(float(self.curriculum_level))
				self.trainer.noise.set_sigma(float(self.ounoise_sigma))
			return True
		return False

	def log_scalar(self, name: str, value: float, step: int) -> None:
		with self.writer.as_default():
			tf.summary.scalar(name, value, step=step)
