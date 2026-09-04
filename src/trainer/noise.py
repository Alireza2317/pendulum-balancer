import numpy as np

from src.config import Config


class OUNoise:
	"""Ornstein-Uhlenbeck process for temporally correlated action exploration."""

	def __init__(self, config: Config) -> None:
		self.cfg = config
		self._sigma: float = self.cfg.ounoise_sigma
		self.reset()

	def reset(self) -> None:
		"""Reset internal state to mean (to be called at the start of every episode)."""
		self.state = np.ones(self.cfg.action_dim) * self.cfg.ounoise_mu

	@property
	def sigma(self) -> float:
		return self._sigma

	def set_sigma(self, value: float) -> None:
		self._sigma = value

	def decay(self, min_sigma: float | None = None) -> None:
		"""
		Anneals exploration magnitude toward ounoise_sigma_min. To be Called once
		per episode."""
		floor: float = self.cfg.ounoise_sigma_min if min_sigma is None else min_sigma
		self._sigma = max(floor, self._sigma * self.cfg.ounoise_decay)

	def sample(self) -> float:
		"""Generate correlated noise step."""
		dx = self.cfg.ounoise_theta * (
			self.cfg.ounoise_mu - self.state
		) + self._sigma * np.random.randn(self.cfg.action_dim)
		self.state += dx
		return float(self.state[0])
