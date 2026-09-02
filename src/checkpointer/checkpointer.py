from pathlib import Path

import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.trainer.curriculum import CurriculumManager


class ModelCheckpointer:
	def __init__(
		self,
		agent: DDPGAgent,
		log_dir: Path | str = Path("logs"),
		checkpoint_dir: Path | str = Path("checkpoints"),
		max_to_keep: int = 3,
		curriculum: CurriculumManager | None = None,
	) -> None:
		self.log_dir = Path(log_dir)
		self.checkpoint_dir = Path(checkpoint_dir)
		self.curriculum = curriculum

		self.episode_counter = tf.Variable(0, dtype=tf.uint64)

		# Persists CurriculumManager.level across process restarts.
		self.curriculum_level = tf.Variable(0.0, dtype=tf.float32)

		self.writer = tf.summary.create_file_writer(str(self.log_dir))
		self.checkpoint = tf.train.Checkpoint(
			actor=agent.actor,
			critic=agent.critic,
			target_actor=agent.target_actor,
			target_critic=agent.target_critic,
			episode=self.episode_counter,
			curriculum_level=self.curriculum_level,
		)
		self.manager = tf.train.CheckpointManager(
			self.checkpoint, directory=str(self.checkpoint_dir), max_to_keep=max_to_keep
		)

	def save(self, step: int) -> str | None:
		if self.curriculum is not None:
			self.curriculum_level.assign(self.curriculum.level)
		return self.manager.save(checkpoint_number=step)

	def load_latest(self) -> bool:
		if self.manager.latest_checkpoint:
			self.checkpoint.restore(self.manager.latest_checkpoint)
			if self.curriculum is not None:
				self.curriculum.set_level(float(self.curriculum_level))
			return True
		return False

	def log_scalar(self, name: str, value: float, step: int) -> None:
		with self.writer.as_default():
			tf.summary.scalar(name, value, step=step)
