import argparse
import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.agent.memory import UniformReplayBuffer
from src.config import Config
from src.physics.env import DoublePendulumEnv
from src.physics.state import EnvState
from src.trainer.curriculum import CurriculumManager, DifficultyParams
from src.trainer.evaluation import EpisodeResult
from src.trainer.trainer import DDPGTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "artifacts" / "evaluations"
SUCCESS_BALANCE_FRACTION = 0.8


class EvalMode(Enum):
	FULL = "full"
	SENTINEL = "sentinel"


@dataclass(frozen=True)
class EvalCase:
	pole1_angle_deg: float
	pole2_angle_deg: float


@dataclass(frozen=True)
class CaseEvaluationResult:
	case_: EvalCase
	result: EpisodeResult


@dataclass(frozen=True)
class GridEvaluationResult:
	difficulty: DifficultyParams
	mode: EvalMode
	max_steps: int
	case_results: tuple[CaseEvaluationResult, ...]
	n_successful_cases: int
	mean_balance_fraction: float
	worst_balance_fraction: float
	mean_survival_fraction: float


def generate_angle_cases(
	difficulty: DifficultyParams, mode: EvalMode = EvalMode.SENTINEL, grid_size: int = 5
) -> list[EvalCase]:
	"""
	Generate deterministic initial pole-angle cases.

	FULL creates an evenly spaced square grid. Its size must be odd so the
	grid always includes the exact center point.

	SENTINEL creates the center and four corners of the difficulty range.
	"""
	angle_range: float = difficulty.reset_angle_range_deg
	if mode == EvalMode.FULL:
		if grid_size < 3 or grid_size % 2 == 0:
			raise ValueError("grid_size must be an odd integer (>= 3).")

	if mode == EvalMode.FULL:
		angles = np.linspace(-angle_range, angle_range, num=grid_size)

		return [EvalCase(float(a1), float(a2)) for a1 in angles for a2 in angles]

	if mode == EvalMode.SENTINEL:
		return [
			EvalCase(-angle_range, -angle_range),
			EvalCase(-angle_range, angle_range),
			EvalCase(0, 0),
			EvalCase(angle_range, -angle_range),
			EvalCase(angle_range, angle_range),
		]


def is_successful(result: EpisodeResult, max_steps: int) -> bool:
	"""Return whether an episode satisfies the evaluation success criteria."""

	return (
		result.steps == max_steps
		and result.failure_reason is None
		and result.balance_fraction >= SUCCESS_BALANCE_FRACTION
	)


def evaluate_angle_grid(
	trainer: DDPGTrainer,
	difficulty: DifficultyParams,
	mode: EvalMode = EvalMode.SENTINEL,
	grid_size: int = 5,
) -> GridEvaluationResult:
	"""Evaluate the trainer's current actor on a deterministic angle grid."""

	cases: list[EvalCase] = generate_angle_cases(difficulty, mode, grid_size)

	trainer.env.set_difficulty(difficulty.reset_angle_range_deg)

	case_results: list[CaseEvaluationResult] = []
	for case_ in cases:
		state: EnvState = EnvState(
			cart_x=0.0,
			cart_x_velocity=0.0,
			pole1_angle=float(np.deg2rad(case_.pole1_angle_deg)),
			pole1_angular_velocity=0.0,
			pole2_angle=float(np.deg2rad(case_.pole2_angle_deg)),
			pole2_angular_velocity=0.0,
		)

		episode_result: EpisodeResult = trainer.evaluate_episode(state)

		case_results.append(CaseEvaluationResult(case_, episode_result))

	max_steps = trainer.cfg.max_episode_steps

	balance_fractions = [
		case_result.result.balance_fraction for case_result in case_results
	]

	survival_fractions = [
		min(1.0, case_result.result.steps / max_steps) for case_result in case_results
	]

	n_successful_cases = sum(
		is_successful(case_result.result, max_steps) for case_result in case_results
	)

	return GridEvaluationResult(
		difficulty=difficulty,
		mode=mode,
		max_steps=max_steps,
		case_results=tuple(case_results),
		n_successful_cases=n_successful_cases,
		mean_balance_fraction=float(np.mean(balance_fractions)),
		worst_balance_fraction=min(balance_fractions),
		mean_survival_fraction=float(np.mean(survival_fractions)),
	)


def dump_grid_results(
	result: GridEvaluationResult,
	output_path: Path | str | None = None,
) -> Path:
	"""Write all per-case evaluation results to CSV and return its path."""
	if output_path is None:
		level = f"{result.difficulty.level:g}"
		filename = f"grid_level_{level}_{result.mode.value}.csv"
		output_path = DEFAULT_RESULTS_DIR / filename

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	fieldnames = (
		"level",
		"reset_angle_range_deg",
		"mode",
		"pole1_angle_deg",
		"pole2_angle_deg",
		"total_reward",
		"steps",
		"balance_fraction",
		"failure_reason",
		"successful",
		"survival_fraction",
	)

	with output_path.open("w", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()

		for case_result in result.case_results:
			episode = case_result.result

			writer.writerow(
				{
					"level": result.difficulty.level,
					"reset_angle_range_deg": (result.difficulty.reset_angle_range_deg),
					"mode": result.mode.value,
					"pole1_angle_deg": case_result.case_.pole1_angle_deg,
					"pole2_angle_deg": case_result.case_.pole2_angle_deg,
					"total_reward": episode.total_reward,
					"steps": episode.steps,
					"balance_fraction": episode.balance_fraction,
					"failure_reason": episode.failure_reason,
					"successful": is_successful(
						episode,
						result.max_steps,
					),
					"survival_fraction": min(
						1.0,
						episode.steps / result.max_steps,
					),
				}
			)

	return output_path


def curriculum_level_argument(value: str) -> float:
	"""Parse and validate a curriculum level supplied on the command line."""
	level = float(value)

	if not 0.0 <= level <= 1.0:
		raise argparse.ArgumentTypeError("level must be between 0.0 and 1.0")

	return level


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate a saved actor on a deterministic angle grid."
	)

	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=None,
		help=(
			"Checkpoint prefix, such as checkpoints/ckpt-1900. "
			"If omitted, the latest checkpoint is used."
		),
	)

	parser.add_argument(
		"--mode",
		choices=[mode.value for mode in EvalMode],
		default=EvalMode.FULL.value,
		help="Evaluation case set to run.",
	)

	parser.add_argument(
		"--level",
		type=curriculum_level_argument,
		default=0.0,
		help="Curriculum level between 0.0 and 1.0.",
	)

	parser.add_argument(
		"--grid-size",
		type=int,
		default=5,
		help="Number of points per angle axis in full-grid mode.",
	)

	return parser.parse_args()


def resolve_checkpoint(
	checkpoint_argument: Path | None,
) -> str:
	"""Resolve an explicit checkpoint or find the latest saved checkpoint."""
	checkpoint_directory = PROJECT_ROOT / "checkpoints"

	if checkpoint_argument is None:
		checkpoint = tf.train.latest_checkpoint(str(checkpoint_directory))

		if checkpoint is None:
			raise FileNotFoundError(f"No checkpoint found in {checkpoint_directory}")

		return checkpoint

	checkpoint_path = checkpoint_argument

	if not checkpoint_path.is_absolute():
		checkpoint_path = PROJECT_ROOT / checkpoint_path

	checkpoint_index = Path(f"{checkpoint_path}.index")

	if not checkpoint_index.is_file():
		raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

	return str(checkpoint_path)


def main():
	args = parse_arguments()
	cfg = Config()

	checkpoint = resolve_checkpoint(args.checkpoint)
	mode = EvalMode(args.mode)

	curriculum = CurriculumManager(cfg)
	curriculum.set_level(args.level)
	difficulty = curriculum.current_params()

	agent = DDPGAgent(cfg)
	tf.train.Checkpoint(actor=agent.actor).restore(checkpoint).expect_partial()

	env = DoublePendulumEnv(cfg, render=False)
	try:
		trainer = DDPGTrainer(cfg, env, agent, UniformReplayBuffer(1))

		print(
			f"Evaluating {Path(checkpoint).name} at "
			f"level {difficulty.level:g} in {mode.value} mode..."
		)

		grid_result = evaluate_angle_grid(
			trainer=trainer,
			difficulty=difficulty,
			mode=mode,
			grid_size=args.grid_size,
		)

		checkpoint_name = Path(checkpoint).name
		output_filename = (
			f"{checkpoint_name}_level_{difficulty.level:g}_{mode.value}.csv"
		)
		output_path = DEFAULT_RESULTS_DIR / output_filename

		dump_grid_results(
			grid_result,
			output_path=output_path,
		)

		print(
			f"Summary: {grid_result.n_successful_cases}/"
			+ f"{len(grid_result.case_results)} successful, "
			+ f"mean balance={grid_result.mean_balance_fraction:.3f}, "
			+ f"worst balance={grid_result.worst_balance_fraction:.3f}, "
			+ f"mean survival={grid_result.mean_survival_fraction:.3f}"
		)
		print(f"Saved detailed results to {output_path}")
	finally:
		env.close()


if __name__ == "__main__":
	main()
