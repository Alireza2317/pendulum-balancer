from pathlib import Path

import tensorflow as tf

from src.agent.agent import DDPGAgent


class ModelCheckpointer:
	def __init__(
		self,
		agent: DDPGAgent,
		log_dir: Path | str = Path("logs"),
		checkpoint_dir: Path | str = Path("checkpoints"),
		max_to_keep: int = 3,
	) -> None:
		self.log_dir = Path(log_dir)
		self.checkpoint_dir = Path(checkpoint_dir)

		self.episode_counter = tf.Variable(0, dtype=tf.uint64)

		self.writer = tf.summary.create_file_writer(str(self.log_dir))
		self.checkpoint = tf.train.Checkpoint(
			actor=agent.actor,
			critic=agent.critic,
			target_actor=agent.target_actor,
			target_critic=agent.target_critic,
			episode=self.episode_counter
		)
		self.manager = tf.train.CheckpointManager(
			self.checkpoint, directory=str(self.checkpoint_dir), max_to_keep=max_to_keep
		)

	def save(self, step: int) -> str | None:
		return self.manager.save(checkpoint_number=step)

	def load_latest(self) -> bool:
		if self.manager.latest_checkpoint:
			self.checkpoint.restore(self.manager.latest_checkpoint)
			return True
		return False

	def log_scalar(self, name: str, value: float, step: int) -> None:
		with self.writer.as_default():
			tf.summary.scalar(name, value, step=step)
