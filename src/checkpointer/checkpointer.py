from pathlib import Path

import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.config import PROJECT_ROOT
from src.trainer.trainer import DDPGTrainer


class ModelCheckpointer:
	def __init__(
		self,
		agent: DDPGAgent,
		trainer: DDPGTrainer | None = None,
		log_dir: Path | str = PROJECT_ROOT / "logs",
		checkpoint_dir: Path | str = PROJECT_ROOT / "checkpoints",
		max_to_keep: int = 50,
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

	def _save_sidefiles(self, step: int) -> None:
		if self.trainer is None:
			return

		window_path, buffer_path = self._sidefile_paths(step)

		# Write temporary files first so an interrupted save cannot corrupt
		# the last valid side files.
		window_temporary = window_path.with_suffix(".pkl.tmp")
		buffer_temporary = buffer_path.with_suffix(".pkl.tmp")

		try:
			self.trainer.curriculum.save(window_temporary)
			self.trainer.buffer.save(buffer_temporary)

			window_temporary.replace(window_path)
			buffer_temporary.replace(buffer_path)
		finally:
			window_temporary.unlink(missing_ok=True)
			buffer_temporary.unlink(missing_ok=True)

	def _load_sidefiles(self, step: int) -> None:
		if self.trainer is None:
			return

		window_path, buffer_path = self._sidefile_paths(step)

		if not window_path.is_file():
			raise FileNotFoundError(
				f"Curriculum window is missing for checkpoint {step}: {window_path}"
			)

		if not buffer_path.is_file():
			raise FileNotFoundError(
				f"Replay buffer is missing for checkpoint {step}: {buffer_path}"
			)

		self.trainer.curriculum.load(window_path)
		self.trainer.buffer.load(buffer_path)

	@staticmethod
	def _checkpoint_step(checkpoint_path: str) -> int:
		"""Extract the checkpoint number from a path such as `ckpt-1000`."""
		name = Path(checkpoint_path).name

		try:
			return int(name.rsplit("-", maxsplit=1)[1])
		except (IndexError, ValueError) as error:
			raise ValueError(
				f"Cannot determine checkpoint step from {checkpoint_path}"
			) from error

	def _prune_sidefiles(self) -> None:
		"""Delete sidefiles whose TensorFlow checkpoints are no longer retained."""
		retained_steps = {
			self._checkpoint_step(checkpoint_path)
			for checkpoint_path in self.manager.checkpoints
		}

		for pattern in ("window_*.pkl", "buffer_*.pkl"):
			for path in self.checkpoint_dir.glob(pattern):
				try:
					step = int(path.stem.rsplit("_", maxsplit=1)[1])
				except (IndexError, ValueError):
					continue

				if step not in retained_steps:
					path.unlink()

	def save(self, step: int) -> str:
		self.episode_counter.assign(step)
		if self.trainer is not None:
			self.curriculum_level.assign(self.trainer.curriculum.level)
			self.ounoise_sigma.assign(self.trainer.noise.sigma)

		# Save window and buffer files.
		self._save_sidefiles(step)

		checkpoint_path = self.manager.save(checkpoint_number=step)

		if checkpoint_path is None:
			# The checkpoint was not published, so its sidefiles should not
			# appear as a valid resumable set.
			window_path, buffer_path = self._sidefile_paths(step)
			window_path.unlink(missing_ok=True)
			buffer_path.unlink(missing_ok=True)

			raise RuntimeError("TensorFlow failed to save the checkpoint")

		# Remove sidefiles that no longer have a dedicated checkpoint.
		self._prune_sidefiles()

		return checkpoint_path

	def _sidefile_paths(self, step: int) -> tuple[Path, Path]:
		return (
			self.checkpoint_dir / f"window_{step}.pkl",
			self.checkpoint_dir / f"buffer_{step}.pkl",
		)

	def load_checkpoint(
		self,
		checkpoint_path: Path | str,
		load_sidefiles: bool = True,
	) -> None:
		"""Restore a specific checkpoint and optionally its matching sidefiles."""
		checkpoint_path = Path(checkpoint_path)
		checkpoint_index = Path(f"{checkpoint_path}.index")

		if not checkpoint_index.is_file():
			raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

		self.checkpoint.restore(str(checkpoint_path))

		if self.trainer is None:
			return

		self.trainer.curriculum.set_level(float(self.curriculum_level))
		self.trainer.noise.set_sigma(float(self.ounoise_sigma))

		if load_sidefiles:
			step = int(self.episode_counter)
			self._load_sidefiles(step)

	def load_latest(self) -> bool:
		checkpoint_path = self.manager.latest_checkpoint
		if checkpoint_path is None:
			return False
		self.load_checkpoint(checkpoint_path, load_sidefiles=self.trainer is not None)

		return True

	def log_scalar(self, name: str, value: float, step: int) -> None:
		with self.writer.as_default():
			tf.summary.scalar(name, value, step=step)
