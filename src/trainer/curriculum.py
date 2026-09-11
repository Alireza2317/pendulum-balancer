import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from src.config import Config


@dataclass(frozen=True)
class DifficultyParams:
	reset_angle_range_deg: float
	angle_threshold_deg: float
	level: float


class CurriculumManager:
	"""
	Increase environment difficulty as the agent demonstrates reliable control.

	Each episode is considered successful when it:
		- reaches the configured maximum number of steps without failing;
		- remains balanced for at least the configured fraction of its steps.

	`success_ratio` is the fraction of successful episodes in the current
	rolling window. When the window is full and its success ratio reaches the
	threshold for the current level, the curriculum advances one step.

	As the level increases, both the random reset-angle range and the allowed
	pole-angle threshold increase. The agent therefore progresses from balancing
	near vertical toward recovering from increasingly large pole angles.
	"""

	def __init__(self, cfg: Config) -> None:
		self.cfg = cfg
		self._level: float = 0.0
		self._window: deque[bool] = deque(maxlen=cfg.curriculum_window)
		self._episodes_since_advance: int = 0

	@property
	def episodes_since_advance(self) -> int:
		return self._episodes_since_advance

	@property
	def level(self) -> float:
		"""Current normalized curriculum level in the range [0, 1]."""
		return self._level

	def set_level(self, level: float) -> None:
		self._level = max(0.0, min(1.0, level))
		# Clears the rolling window so the agent has to re-prove itself at this level
		# before advancing further
		self._window.clear()
		self._episodes_since_advance = 0

	@property
	def success_ratio(self) -> float:
		"""Fraction of successful episodes in the current rolling window."""

		if not self._window:
			return 0.0
		return sum(self._window) / len(self._window)

	def current_success_ratio_threshold(self, level: float) -> float:
		"""
		Return the window success ratio required to advance from `level`.

		The required ratio gradually decreases as the curriculum becomes harder.
		"""
		return self._lerp(
			self.cfg.curriculum_success_ratio,
			self.cfg.curriculum_success_ratio_min,
			level,
		)

	def current_balance_fraction_threshold(self, level: float) -> float:
		return self.cfg.curriculum_episode_balance_threshold

	def record_episode(
		self, steps_performed: int, balance_fraction: float, failed: bool
	) -> None:
		"""
		Call once per completed episode with:
			- Number of steps performed in the episode.
			- Balance fraction (# of balanced steps / # of performed steps).
			- If the episode failed or not.
		"""

		episode_succeeded: bool = (
			not failed
			and steps_performed == self.cfg.max_episode_steps
			and balance_fraction >= self.current_balance_fraction_threshold(self.level)
		)

		self._window.append(episode_succeeded)

		self._episodes_since_advance += 1

	@property
	def ready_to_advance(self) -> bool:
		return (
			self.level < 1.0
			and len(self._window) == self._window.maxlen
			and self.success_ratio >= self.current_success_ratio_threshold(self._level)
		)

	def advance(self) -> None:
		if self.level >= 1.0:
			return

		self._level = min(1.0, self._level + self.cfg.curriculum_step)
		self._window.clear()
		self._episodes_since_advance = 0

	@staticmethod
	def _lerp(a: float, b: float, t: float) -> float:
		return a + (b - a) * t

	def current_params(self) -> DifficultyParams:
		reset_deg = self._lerp(
			self.cfg.curriculum_reset_start_deg,
			self.cfg.curriculum_reset_end_deg,
			self._level,
		)
		margin_deg = self._lerp(
			self.cfg.curriculum_margin_start_deg,
			self.cfg.curriculum_margin_end_deg,
			self._level,
		)
		# Clamp to 180: beyond that the threshold can never trigger anyway
		# (normalized angle magnitude never exceeds 180 deg), which is exactly
		# the "angle-termination disabled" state we want at max difficulty.
		threshold_deg = min(180.0, reset_deg + margin_deg)
		return DifficultyParams(
			reset_angle_range_deg=reset_deg,
			angle_threshold_deg=threshold_deg,
			level=self._level,
		)

	def save(self, filepath: Path | str) -> None:
		state = {
			"window": self._window,
			"episodes_since_advance": self.episodes_since_advance,
		}

		with open(filepath, "wb") as f:
			pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

	def load(self, filepath: Path | str) -> None:
		with open(filepath, "rb") as f:
			state = pickle.load(f)

		# Backward compatibility
		if isinstance(state, deque):
			self._window = state
			self._episodes_since_advance = len(self._window)
			return

		self._window = state["window"]
		self._episodes_since_advance = state["episodes_since_advance"]
