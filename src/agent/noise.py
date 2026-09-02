import numpy as np

from src.config import Config


class OUNoise:
	"""Ornstein-Uhlenbeck process for temporally correlated action exploration."""

	def __init__(self, config: Config) -> None:
		self.cfg = config
		self.sigma: float = self.cfg.ounoise_sigma
		self.reset()

	def reset(self) -> None:
		"""Reset internal state to mean (to be called at the start of every episode)."""
		self.state = np.ones(self.cfg.action_dim) * self.cfg.ounoise_mu

	def decay(self) -> None:
		"""
		Anneals exploration magnitude toward ounoise_sigma_min. To be Called once
		per episode."""
		self.sigma = max(
			self.cfg.ounoise_sigma_min, self.sigma * self.cfg.ounoise_decay
		)

	def sample(self) -> float:
		"""Generate correlated noise step."""
		dx = self.cfg.ounoise_theta * (
			self.cfg.ounoise_mu - self.state
		) + self.cfg.ounoise_sigma * np.random.randn(self.cfg.action_dim)
		self.state += dx
		return float(self.state[0])
