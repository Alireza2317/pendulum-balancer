import math

from src.config import Config
from src.trainer.noise import OUNoise


class ExplorationScheduler:
	def __init__(self, cfg: Config, noise: OUNoise) -> None:
		self.cfg = cfg
		self.noise: OUNoise = noise
		self._last_seen_level: float = 0

		self._exploration_decay_active: bool = False

	def on_episode_start(
		self, level: float, success_ratio: float, episodes_since_advance: int
	) -> None:
		self.noise.reset()

		leveled_up: bool = not math.isclose(level, self._last_seen_level)
		self._last_seen_level = level

		if success_ratio >= self.cfg.exploration_decay_unlock_threshold:
			self._exploration_decay_active = True

		if not self._exploration_decay_active:
			return

		min_sigma: float = self._min_sigma_for_level(level)

		stagnant: bool = (
			episodes_since_advance > 0
			and episodes_since_advance % self.cfg.stagnation_reheat_interval == 0
		)

		if leveled_up:
			# Set sigma to its new value
			self.noise.set_sigma(self._sigma_for_new_level(level, min_sigma))
		elif stagnant:
			# Reheat as if we'd advanced one step
			pseudo_level = min(1.0, level + self.cfg.curriculum_step)
			min_sigma = self._min_sigma_for_level(pseudo_level)
			self.noise.set_sigma(self._sigma_for_new_level(pseudo_level, min_sigma))
		else:
			# Normal decay, based on previous values
			self.noise.decay(min_sigma)

	@staticmethod
	def _lerp(a: float, b: float, t: float) -> float:
		return a + (b - a) * t

	def _sigma_for_new_level(self, level: float, min_sigma: float) -> float:
		"""
		Sigma to reintroduce when the curriculum advances to `level`.
		Interpolates from the minimum noise at level 0 to the maximum noise at level 1.
		Lower levels mostly need refinement of existing skill, higher levels need real
		exploration to discover recovery/swing-up behavior. It gives stronger bumps at
		early levels. It never decreases sigma on a level-up.
		"""
		reset_sigma = self._lerp(min_sigma, self.cfg.ounoise_sigma, math.sqrt(level))

		return max(reset_sigma, self.noise.sigma)

	def _min_sigma_for_level(self, level: float) -> float:
		"""
		Resting noise floor for the current curriculum level. Interpolates from
		ounoise_sigma_min at level 0 to ounoise_sigma_min_max at level 1. Harder levels
		(recovery/swing-up) plausibly still need occasional large corrective pushes even
		once "converged", unlike fine balance.
		"""
		return self._lerp(
			self.cfg.ounoise_sigma_min, self.cfg.ounoise_sigma_min_max, level
		)
