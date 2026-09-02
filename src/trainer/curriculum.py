from collections import deque
from dataclasses import dataclass

from src.config import Config


@dataclass(frozen=True)
class DifficultyParams:
	reset_angle_range_deg: float
	angle_threshold_deg: float
	level: float


class CurriculumManager:
	"""
	Grows episode difficulty from "balance near vertical" to "recover from a full fall"
	as the agent's performance improves.

	`reset_angle_range_deg` (how far from vertical each pole starts) and
	`angle_threshold_deg` (the angle at which the episode fails) are increased together,
	with `angle_threshold_deg` always kept a margin above the reset	range. That margin
	itself grows over time, so early on the agent mostly just has to hold near vertical,
	but eventually the margin is large enough that angle alone won't end the episode.
	At that point the pole can be reset fully hanging down (180 deg) and the only way to
	get reward is to swing back up and hold it.

	Progression is driven by a rolling average of steps-survived-fraction over
	`cfg.curriculum_window`(e.g. 50) episodes: once that average clears
	`cfg.curriculum_success_ratio`, the level is bumped by `cfg.curriculum_step`and the
	window is cleared so the agent must re-prove itself at the new difficulty before
	advancing further.
	"""

	def __init__(self, cfg: Config) -> None:
		self.cfg = cfg
		self._level: float = 0.0
		self._window: deque[float] = deque(maxlen=cfg.curriculum_window)

	@property
	def level(self) -> float:
		return self._level

	@property
	def success_ratio(self) -> float:
		if not self._window:
			return 0.0
		return sum(self._window) / len(self._window)

	def record_episode(self, steps_survived: int) -> None:
		"""Call once per completed episode with the number of steps it lasted."""
		fraction = min(1.0, steps_survived / self.cfg.max_episode_steps)
		self._window.append(fraction)

		if (
			len(self._window) == self._window.maxlen
			and self.success_ratio >= self.cfg.curriculum_success_ratio
			and self._level < 1.0
		):
			self._level = min(1.0, self._level + self.cfg.curriculum_step)
			self._window.clear()

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
