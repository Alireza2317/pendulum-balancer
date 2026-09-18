import csv
import math
import tempfile
import unittest
from pathlib import Path
from typing import cast

from src.evaluation.grid import (
	CaseEvaluationResult,
	EvalCase,
	EvalMode,
	GridEvaluationResult,
	dump_grid_results,
	evaluate_angle_grid,
	generate_angle_cases,
	is_successful,
)
from src.physics.state import EnvState
from src.trainer.curriculum import DifficultyParams
from src.trainer.evaluation import EpisodeResult
from src.trainer.trainer import DDPGTrainer


class FakeEnvironment:
	def __init__(self) -> None:
		self.difficulty: float | None = None

	def set_difficulty(
		self,
		reset_angle_range_deg: float,
	) -> None:
		self.difficulty = reset_angle_range_deg


class FakeConfig:
	def __init__(self, max_episode_steps: int) -> None:
		self.max_episode_steps = max_episode_steps


class FakeTrainer:
	def __init__(
		self,
		results: list[EpisodeResult],
		max_episode_steps: int,
	) -> None:
		self.cfg = FakeConfig(max_episode_steps)
		self.env = FakeEnvironment()
		self.initial_states: list[EnvState] = []
		self._results = iter(results)

	def evaluate_episode(
		self,
		initial_state: EnvState | None = None,
	) -> EpisodeResult:
		if initial_state is None:
			raise AssertionError(
				"Grid evaluation must supply an explicit initial state"
			)

		self.initial_states.append(initial_state)
		return next(self._results)


class AngleCaseGenerationTests(unittest.TestCase):
	def setUp(self) -> None:
		self.difficulty = DifficultyParams(
			reset_angle_range_deg=6.0,
			level=0.0,
		)

	def test_sentinel_contains_center_and_four_corners(self) -> None:
		cases = generate_angle_cases(
			self.difficulty,
			mode=EvalMode.SENTINEL,
		)

		self.assertEqual(
			cases,
			[
				EvalCase(-6.0, -6.0),
				EvalCase(-6.0, 6.0),
				EvalCase(0.0, 0.0),
				EvalCase(6.0, -6.0),
				EvalCase(6.0, 6.0),
			],
		)

	def test_full_grid_contains_all_combinations(self) -> None:
		cases = generate_angle_cases(
			self.difficulty,
			mode=EvalMode.FULL,
			grid_size=5,
		)

		expected_angles = {-6.0, -3.0, 0.0, 3.0, 6.0}

		self.assertEqual(len(cases), 25)
		self.assertEqual(len(set(cases)), 25)
		self.assertEqual(
			{case.pole1_angle_deg for case in cases},
			expected_angles,
		)
		self.assertEqual(
			{case.pole2_angle_deg for case in cases},
			expected_angles,
		)
		self.assertIn(EvalCase(0.0, 0.0), cases)

	def test_full_grid_rejects_even_size(self) -> None:
		with self.assertRaises(ValueError):
			generate_angle_cases(
				self.difficulty,
				mode=EvalMode.FULL,
				grid_size=4,
			)

	def test_full_grid_rejects_size_smaller_than_three(self) -> None:
		with self.assertRaises(ValueError):
			generate_angle_cases(
				self.difficulty,
				mode=EvalMode.FULL,
				grid_size=1,
			)


class SuccessCriteriaTests(unittest.TestCase):
	def test_complete_balanced_episode_is_successful(self) -> None:
		result = EpisodeResult(
			total_reward=50.0,
			steps=10,
			balance_fraction=0.8,
			failure_reason=None,
		)

		self.assertTrue(is_successful(result, max_steps=10))

	def test_low_balance_episode_is_not_successful(self) -> None:
		result = EpisodeResult(
			total_reward=50.0,
			steps=10,
			balance_fraction=0.79,
			failure_reason=None,
		)

		self.assertFalse(is_successful(result, max_steps=10))

	def test_early_failure_is_not_successful(self) -> None:
		result = EpisodeResult(
			total_reward=20.0,
			steps=4,
			balance_fraction=1.0,
			failure_reason="cart",
		)

		self.assertFalse(is_successful(result, max_steps=10))

	def test_terminal_reason_prevents_success(self) -> None:
		result = EpisodeResult(
			total_reward=50.0,
			steps=10,
			balance_fraction=1.0,
			failure_reason="pole",
		)

		self.assertFalse(is_successful(result, max_steps=10))


class GridEvaluationTests(unittest.TestCase):
	def setUp(self) -> None:
		self.difficulty = DifficultyParams(
			reset_angle_range_deg=2.0,
			level=0.0,
		)

	def test_evaluation_applies_difficulty_and_calculates_summary(
		self,
	) -> None:
		episode_results = [
			EpisodeResult(50.0, 10, 0.9, None),
			EpisodeResult(45.0, 10, 0.7, None),
			EpisodeResult(15.0, 4, 0.5, "cart"),
			EpisodeResult(55.0, 10, 1.0, None),
			EpisodeResult(48.0, 10, 0.8, None),
		]

		fake_trainer = FakeTrainer(
			results=episode_results,
			max_episode_steps=10,
		)

		result = evaluate_angle_grid(
			trainer=cast(DDPGTrainer, fake_trainer),
			difficulty=self.difficulty,
			mode=EvalMode.SENTINEL,
		)

		self.assertEqual(
			fake_trainer.env.difficulty,
			2.0,
		)
		self.assertEqual(len(fake_trainer.initial_states), 5)
		self.assertIsInstance(result.case_results, tuple)
		self.assertEqual(result.n_successful_cases, 3)
		self.assertAlmostEqual(result.mean_balance_fraction, 0.78)
		self.assertAlmostEqual(result.worst_balance_fraction, 0.5)
		self.assertAlmostEqual(result.mean_survival_fraction, 0.88)

	def test_evaluation_converts_case_angles_to_radians(self) -> None:
		episode_results = [EpisodeResult(50.0, 10, 1.0, None) for _ in range(5)]
		fake_trainer = FakeTrainer(
			results=episode_results,
			max_episode_steps=10,
		)

		evaluate_angle_grid(
			trainer=cast(DDPGTrainer, fake_trainer),
			difficulty=self.difficulty,
			mode=EvalMode.SENTINEL,
		)

		first_state = fake_trainer.initial_states[0]

		self.assertAlmostEqual(
			first_state.pole1_angle,
			math.radians(-2.0),
		)
		self.assertAlmostEqual(
			first_state.pole2_angle,
			math.radians(-2.0),
		)
		self.assertEqual(first_state.cart_x, 0.0)
		self.assertEqual(first_state.cart_x_velocity, 0.0)
		self.assertEqual(first_state.pole1_angular_velocity, 0.0)
		self.assertEqual(first_state.pole2_angular_velocity, 0.0)


class GridResultDumpTests(unittest.TestCase):
	def test_dump_writes_flattened_case_rows(self) -> None:
		difficulty = DifficultyParams(
			reset_angle_range_deg=5.0,
			level=0.0,
		)
		episode = EpisodeResult(
			total_reward=123.5,
			steps=4,
			balance_fraction=0.5,
			failure_reason="cart",
		)
		result = GridEvaluationResult(
			difficulty=difficulty,
			mode=EvalMode.SENTINEL,
			max_steps=10,
			case_results=(
				CaseEvaluationResult(
					case_=EvalCase(-5.0, 5.0),
					result=episode,
				),
			),
			n_successful_cases=0,
			mean_balance_fraction=0.5,
			worst_balance_fraction=0.5,
			mean_survival_fraction=0.4,
		)

		with tempfile.TemporaryDirectory() as directory:
			output = Path(directory) / "results.csv"

			returned_path = dump_grid_results(
				result,
				output_path=output,
			)

			self.assertEqual(returned_path, output)
			self.assertTrue(output.exists())

			with output.open(newline="") as file:
				rows = list(csv.DictReader(file))

		self.assertEqual(len(rows), 1)

		row = rows[0]
		self.assertEqual(row["mode"], "sentinel")
		self.assertEqual(float(row["pole1_angle_deg"]), -5.0)
		self.assertEqual(float(row["pole2_angle_deg"]), 5.0)
		self.assertEqual(float(row["total_reward"]), 123.5)
		self.assertEqual(int(row["steps"]), 4)
		self.assertEqual(float(row["balance_fraction"]), 0.5)
		self.assertEqual(row["failure_reason"], "cart")
		self.assertEqual(row["successful"], "False")
		self.assertAlmostEqual(
			float(row["survival_fraction"]),
			0.4,
		)


if __name__ == "__main__":
	unittest.main()
