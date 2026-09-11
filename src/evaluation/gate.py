from dataclasses import dataclass

from src.config import Config
from src.evaluation.grid import (
	EvalMode,
	GridEvaluationResult,
	evaluate_angle_grid,
)
from src.trainer.curriculum import CurriculumManager, DifficultyParams
from src.trainer.trainer import DDPGTrainer


@dataclass(frozen=True)
class GateResult:
	base: GridEvaluationResult
	current: GridEvaluationResult
	advancement_allowed: bool
	regression_detected: bool

	@property
	def score(self) -> tuple[float, int, float, float, int, float]:
		"""
		Rank valid checkpoints.

		Curriculum level comes first, followed by current-level coverage and
		quality, then retained base-level performance.
		"""
		return (
			self.current.difficulty.level,
			self.current.n_successful_cases,
			self.current.mean_survival_fraction,
			self.current.mean_balance_fraction,
			self.base.n_successful_cases,
			self.base.mean_balance_fraction,
		)


def difficulty_at_level(config: Config, level: float) -> DifficultyParams:
	curriculum = CurriculumManager(config)
	curriculum.set_level(level)
	return curriculum.current_params()


def run_evaluation_gate(
	trainer: DDPGTrainer,
	current_level: float,
) -> GateResult:
	"""Evaluate retained base skill and current curriculum mastery."""

	config = trainer.cfg

	base = evaluate_angle_grid(
		trainer,
		difficulty_at_level(config, 0.0),
		mode=EvalMode.SENTINEL,
	)

	if current_level == 0.0:
		current = base
	else:
		current = evaluate_angle_grid(
			trainer,
			difficulty_at_level(config, current_level),
			mode=EvalMode.SENTINEL,
		)

	regression_detected = (
		base.n_successful_cases < config.regression_base_min_successes
		or base.mean_survival_fraction < config.regression_base_min_survival
	)

	advancement_allowed = (
		not regression_detected
		and base.n_successful_cases >= config.advancement_base_successes
		and current.n_successful_cases >= config.advancement_current_successes
		and current.mean_survival_fraction >= config.advancement_current_survival_rate
	)

	return GateResult(
		base=base,
		current=current,
		advancement_allowed=advancement_allowed,
		regression_detected=regression_detected,
	)
