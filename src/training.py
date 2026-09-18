import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.agent.memory import UniformReplayBuffer
from src.checkpointer.checkpointer import ModelCheckpointer
from src.config import PROJECT_ROOT, Config
from src.evaluation.gate import GateResult, run_evaluation_gate
from src.evaluation.grid import dump_grid_results
from src.physics.env import DoublePendulumEnv
from src.trainer.trainer import DDPGTrainer, ResetMode


def resolve_checkpoint(path: Path) -> Path:
	if not path.is_absolute():
		path = PROJECT_ROOT / path

	if not Path(f"{path}.index").is_file():
		raise FileNotFoundError(f"Checkpoint does not exist: {path}")

	return path


def load_best_score(best_directory: Path) -> tuple | None:
	metadata_path = best_directory / "best.json"

	if not metadata_path.is_file():
		return None

	metadata = json.loads(metadata_path.read_text())
	return tuple(metadata["score"])


def log_gate(
	checkpointer: ModelCheckpointer,
	gate: GateResult,
	episode: int,
) -> None:
	checkpointer.log_scalar(
		"Evaluation/Base Successful Cases",
		gate.base.n_successful_cases,
		episode,
	)
	checkpointer.log_scalar(
		"Evaluation/Base Mean Balance",
		gate.base.mean_balance_fraction,
		episode,
	)
	checkpointer.log_scalar(
		"Evaluation/Base Mean Survival",
		gate.base.mean_survival_fraction,
		episode,
	)
	checkpointer.log_scalar(
		"Evaluation/Current Successful Cases",
		gate.current.n_successful_cases,
		episode,
	)
	checkpointer.log_scalar(
		"Evaluation/Current Mean Balance",
		gate.current.mean_balance_fraction,
		episode,
	)
	checkpointer.log_scalar(
		"Evaluation/Current Mean Survival",
		gate.current.mean_survival_fraction,
		episode,
	)


def save_gate_results(
	gate: GateResult,
	evaluation_directory: Path,
	episode: int,
) -> None:
	base_path = evaluation_directory / f"ckpt-{episode}_level_0_sentinel.csv"
	current_level = f"{gate.current.difficulty.level:g}"
	current_path = (
		evaluation_directory / f"ckpt-{episode}_level_{current_level}_sentinel.csv"
	)

	dump_grid_results(gate.base, base_path)

	if current_path != base_path:
		dump_grid_results(gate.current, current_path)


def train(
	*,
	episodes: int,
	run_name: str,
	resume: Path | None = None,
	reset_replay_buffer: bool = False,
) -> None:
	config = Config()
	random.seed(config.seed)
	np.random.seed(config.seed)
	tf.random.set_seed(config.seed)

	if episodes <= 0:
		raise ValueError("--episodes must be positive")

	run_directory = PROJECT_ROOT / "runs" / run_name
	checkpoint_directory = run_directory / "checkpoints"
	best_directory = run_directory / "best"
	evaluation_directory = run_directory / "evaluations"
	log_directory = run_directory / "logs"

	run_directory.mkdir(parents=True, exist_ok=True)
	config.save(run_directory / "config.json")

	manifest = {
		"run_name": run_name,
		"source_checkpoint": (str(resume) if resume is not None else None),
		"episodes_requested": episodes,
		"random_seed": config.seed,
	}
	(run_directory / "manifest.json").write_text(json.dumps(manifest, indent=4))

	env = DoublePendulumEnv(config, render=False)

	try:
		buffer = UniformReplayBuffer(config.buffer_maxsize)
		agent = DDPGAgent(config)
		trainer = DDPGTrainer(config, env, agent, buffer)

		checkpointer = ModelCheckpointer(
			agent,
			trainer,
			log_dir=log_directory,
			checkpoint_dir=checkpoint_directory,
		)

		if resume is None:
			start_episode = 1
		else:
			source_checkpoint = resolve_checkpoint(resume)
			checkpointer.load_checkpoint(source_checkpoint)
			start_episode = int(checkpointer.episode_counter) + 1

			if reset_replay_buffer:
				trainer.buffer.clear()

		actual_actor_lr = float(agent.actor_optimizer.learning_rate.numpy())
		actual_critic_lr = float(agent.critic_optimizer.learning_rate.numpy())

		if not np.isclose(actual_actor_lr, config.actor_lr):
			raise RuntimeError(
				f"Actor learning rate is {actual_actor_lr}, expected {config.actor_lr}"
			)

		if not np.isclose(actual_critic_lr, config.critic_lr):
			raise RuntimeError(
				f"Critic learning rate is {actual_critic_lr}, "
				f"expected {config.critic_lr}"
			)

		print(f"Actor learning rate:  {actual_actor_lr}")
		print(f"Critic learning rate: {actual_critic_lr}")

		best_score = load_best_score(best_directory)
		last_gate_episode: int | None = None
		end_episode = start_episode + episodes - 1
		consecutive_regressions = 0

		for episode in range(start_episode, end_episode + 1):
			print(f"Episode {episode}, level={trainer.curriculum.level:g}")

			result, metrics, difficulty = trainer.run_episode(episode)
			reset_info = trainer.last_reset_info

			if reset_info is None:
				raise RuntimeError("Missing reset information")

			checkpointer.log_scalar(
				"Episode Total Reward",
				result.total_reward,
				episode,
			)
			checkpointer.log_scalar(
				"Steps Survived in Each Episode",
				result.steps,
				episode,
			)
			checkpointer.log_scalar(
				"Balance fraction",
				result.balance_fraction,
				episode,
			)
			checkpointer.log_scalar(
				"Episode Average Actor Loss",
				metrics.actor_loss,
				episode,
			)
			checkpointer.log_scalar(
				"Episode Average Critic Loss",
				metrics.critic_loss,
				episode,
			)
			checkpointer.log_scalar(
				"Episode Average Q-Values",
				metrics.q_vals_avg,
				episode,
			)
			checkpointer.log_scalar(
				"Difficulty Level",
				difficulty.level,
				episode,
			)
			checkpointer.log_scalar(
				"Curriculum Success Ratio",
				trainer.curriculum.success_ratio,
				episode,
			)
			checkpointer.log_scalar(
				"Exploration Noise Sigma",
				trainer.noise.sigma,
				episode,
			)
			checkpointer.log_scalar(
				"Reset/Pole 1 Angle Degrees",
				reset_info.pole1_angle_deg,
				episode,
			)
			checkpointer.log_scalar(
				"Reset/Pole 2 Angle Degrees",
				reset_info.pole2_angle_deg,
				episode,
			)

			reset_mode_number = {
				ResetMode.UNIFORM: 0,
				ResetMode.OPPOSING_POSITIVE_FIRST: 1,
				ResetMode.OPPOSING_NEGATIVE_FIRST: 2,
			}[reset_info.mode]

			checkpointer.log_scalar(
				"Reset/Mode",
				reset_mode_number,
				episode,
			)

			regular_evaluation = episode % config.evaluate_every_n_episodes == 0
			gate_retry_allowed = (
				last_gate_episode is None
				or episode - last_gate_episode >= config.evaluate_gate_retry_episodes
			)
			advancement_evaluation = (
				trainer.curriculum.ready_to_advance and gate_retry_allowed
			)
			final_evaluation = episode == end_episode

			should_evaluate = (
				regular_evaluation or advancement_evaluation or final_evaluation
			)

			gate: GateResult | None = None

			if should_evaluate:
				evaluated_level = trainer.curriculum.level
				gate = run_evaluation_gate(
					trainer,
					current_level=evaluated_level,
				)
				last_gate_episode = episode

				log_gate(checkpointer, gate, episode)
				save_gate_results(
					gate,
					evaluation_directory,
					episode,
				)

				print(
					f"Evaluation: base="
					f"{gate.base.n_successful_cases}/5, "
					f"current="
					f"{gate.current.n_successful_cases}/5, "
					f"current survival="
					f"{gate.current.mean_survival_fraction:.3f}"
				)

				if trainer.curriculum.ready_to_advance and gate.advancement_allowed:
					trainer.curriculum.advance()
					print(f"Curriculum advanced to {trainer.curriculum.level:g}")

			should_save = (
				episode % config.checkpoint_every_n_episodes == 0
				or should_evaluate
				or episode == end_episode
			)

			checkpoint_path: str | None = None

			if should_save:
				checkpoint_path = checkpointer.save(episode)
			if gate is not None and checkpoint_path is not None:
				if gate.regression_detected:
					consecutive_regressions += 1
				else:
					consecutive_regressions = 0

				if not gate.regression_detected:
					if best_score is None or gate.score > best_score:
						best_score = gate.score

						checkpointer.promote_to_best(
							checkpoint_path,
							best_directory,
							gate.score,
							{
								"episode": episode,
								"evaluated_level": (gate.current.difficulty.level),
								"base_successes": (gate.base.n_successful_cases),
								"current_successes": (gate.current.n_successful_cases),
								"current_mean_survival": (
									gate.current.mean_survival_fraction
								),
								"current_mean_balance": (
									gate.current.mean_balance_fraction
								),
							},
						)

						print(f"Promoted checkpoint {episode} as best")

				if consecutive_regressions >= config.regression_patience:
					print(
						"Training stopped after "
						f"{consecutive_regressions} consecutive "
						"base-level regression evaluations."
					)
					break

				if gate.regression_detected:
					print(
						"Regression warning "
						f"({consecutive_regressions}/"
						f"{config.regression_patience}); "
						"the protected best checkpoint is unchanged."
					)

	finally:
		env.close()
