import math

from src.config import Config
from src.trainer.noise import OUNoise


class ExplorationScheduler:
	def __init__(self, cfg: Config, noise: OUNoise) -> None:
		self.cfg = cfg
		self.noise: OUNoise = noise
		self._last_seen_level: float = 0

		self._exploration_decay_active: bool = False

	def on_episode_start(self, level: float, success_ratio: float) -> None:
		self.noise.reset()

		leveled_up: bool = not math.isclose(level, self._last_seen_level)
		self._last_seen_level = level

		if success_ratio >= self.cfg.exploration_decay_unlock_threshold:
			self._exploration_decay_active = True

		if not self._exploration_decay_active:
			return

		if leveled_up:
			# Level-up occured
			# Set sigma to its new value
			self.noise.set_sigma(self._sigma_for_new_level(level))
		else:
			# Normal decay, based on previous values
			self.noise.decay()

	def _sigma_for_new_level(self, level: float) -> float:
		"""
		Sigma to reintroduce when the curriculum advances to `level`.
		Interpolates from a partial reset (min_fraction) at level 0 to a full reset
		(max_fraction) at level 1 -- lower levels mostly need refinement of existing
		skill, higher levels need real exploration to discover recovery/swing-up
		behavior. Never decreases sigma on a level-up.
		"""
		min_fraction = 0.5
		max_fraction = 1.0
		fraction = min_fraction + (max_fraction - min_fraction) * level
		reset_sigma = fraction * self.cfg.ounoise_sigma

		return max(reset_sigma, self.noise.sigma)
