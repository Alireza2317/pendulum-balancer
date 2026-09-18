import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.agent.memory import UniformReplayBuffer
from src.checkpointer.checkpointer import ModelCheckpointer
from src.config import PROJECT_ROOT, Config
from src.evaluation.grid import dump_grid_results, evaluate_angle_grid
from src.physics.env import DoublePendulumEnv
from src.trainer.trainer import DDPGTrainer, ResetMode


def resolve_checkpoint(path: Path) -> Path:
	if not path.is_absolute():
		path = PROJECT_ROOT / path

	if not Path(f"{path}.index").is_file():
		raise FileNotFoundError(f"Checkpoint does not exist: {path}")

	return path


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

		end_episode = start_episode + episodes - 1

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

			if episode % config.evaluate_every_n_episodes == 0:
				evaluation = evaluate_angle_grid(trainer, difficulty)
				for name, value in (
					("Successful Cases", evaluation.n_successful_cases),
					("Mean Balance", evaluation.mean_balance_fraction),
					("Mean Survival", evaluation.mean_survival_fraction),
					("Difficulty Level", difficulty.level),
				):
					checkpointer.log_scalar(f"Evaluation/{name}", value, episode)
				dump_grid_results(
					evaluation,
					run_directory / "evaluations" / f"episode-{episode}_sentinel.csv",
				)
				print(
					f"Evaluation: {evaluation.n_successful_cases}/"
					f"{len(evaluation.case_results)} successful cases, "
					f"mean survival={evaluation.mean_survival_fraction:.3f}"
				)

			if trainer.curriculum.ready_to_advance:
				trainer.curriculum.advance()
				print(f"Curriculum advanced to {trainer.curriculum.level:g}")

			if (
				episode % config.checkpoint_every_n_episodes == 0
				or episode == end_episode
			):
				checkpointer.save(episode)

	finally:
		env.close()
