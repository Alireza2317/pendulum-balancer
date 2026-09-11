import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import random
import time

import numpy as np
import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.agent.memory import UniformReplayBuffer
from src.checkpointer.checkpointer import ModelCheckpointer
from src.config import PROJECT_ROOT, Config
from src.physics.env import DoublePendulumEnv
from src.trainer.trainer import DDPGTrainer, ResetMode

random.seed(23)
np.random.seed(23)
tf.random.set_seed(23)


def train(continue_train: bool = True, save_log_process: bool = True):
	cfg: Config = Config()

	env = DoublePendulumEnv(cfg, render=False)
	try:
		buffer = UniformReplayBuffer(cfg.buffer_maxsize)
		agent = DDPGAgent(cfg)
		trainer = DDPGTrainer(cfg, env, agent, buffer)
		checkpointer = ModelCheckpointer(
			agent, trainer, log_dir=PROJECT_ROOT / "logs" / "lower_actor_lr_from_2406"
		)

		if continue_train:
			# checkpointer.load_latest()
			checkpointer.load_checkpoint(
				PROJECT_ROOT / "checkpoints" / "ckpt-2510",
			)
			# trainer.curriculum.set_level(0.005)
			start_episode: int = int(checkpointer.episode_counter) + 1
			# start_episode: int = 2501
		else:
			start_episode = 1

		actual_actor_lr = float(agent.actor_optimizer.learning_rate.numpy())
		actual_critic_lr = float(agent.critic_optimizer.learning_rate.numpy())

		if not np.isclose(actual_actor_lr, cfg.actor_lr):
			raise RuntimeError(
				f"Actor learning rate is {actual_actor_lr}, expected {cfg.actor_lr}"
			)

		if not np.isclose(actual_critic_lr, cfg.critic_lr):
			raise RuntimeError(
				f"Critic learning rate is {actual_critic_lr}, expected {cfg.critic_lr}"
			)

		for episode in range(start_episode, start_episode + cfg.max_episodes):
			print(f"Running episode {episode:4}...")

			episode_result, agent_metric, difficulty = trainer.run_episode(episode)

			if save_log_process:
				checkpointer.log_scalar(
					"Episode Total Reward", episode_result.total_reward, step=episode
				)
				checkpointer.log_scalar(
					"Steps Survived in Each Episode", episode_result.steps, step=episode
				)
				checkpointer.log_scalar(
					"Balance fraction", episode_result.balance_fraction, step=episode
				)
				checkpointer.log_scalar(
					"Episode Average Actor Loss", agent_metric.actor_loss, step=episode
				)
				checkpointer.log_scalar(
					"Episode Average Critic Loss",
					agent_metric.critic_loss,
					step=episode,
				)
				checkpointer.log_scalar(
					"Episode Average Q-Values", agent_metric.q_vals_avg, step=episode
				)
				checkpointer.log_scalar(
					"Difficulty Level", difficulty.level, step=episode
				)
				checkpointer.log_scalar(
					"Curriculum Success Ratio",
					trainer.curriculum.success_ratio,
					step=episode,
				)
				checkpointer.log_scalar(
					"Exploration Noise Sigma", trainer.noise.sigma, step=episode
				)

				if trainer.last_reset_info is None:
					raise RuntimeError("Trainer did not save any rest information!")

				checkpointer.log_scalar(
					"Reset-pole1-angle",
					trainer.last_reset_info.pole1_angle_deg,
					step=episode,
				)
				checkpointer.log_scalar(
					"Reset-pole2-angle",
					trainer.last_reset_info.pole2_angle_deg,
					step=episode,
				)

				reset_mode_number = {
					ResetMode.UNIFORM: 0,
					ResetMode.OPPOSING_POSITIVE_FIRST: 1,
					ResetMode.OPPOSING_NEGATIVE_FIRST: 2,
				}[trainer.last_reset_info.mode]

				checkpointer.log_scalar(
					"Reset/Mode",
					reset_mode_number,
					step=episode,
				)

			if episode % cfg.checkpoint_every_n_episodes == 0:
				checkpointer.save(episode)
	finally:
		env.close()


def run():
	cfg: Config = Config()

	env = DoublePendulumEnv(cfg, render=True)
	try:
		env.set_difficulty(reset_angle_range_deg=15, angle_threshold_deg=24)
		agent = DDPGAgent(cfg)
		checkpointer = ModelCheckpointer(agent)

		if not checkpointer.load_latest():
			raise FileNotFoundError("No checkpoints found!")

		print("Checkpoint loaded successfully!")

		state = env.reset()
		while True:
			# Get the action from the actor
			action: float = agent.get_action(state, noise=0)

			# Step the environment based on the action
			state, _, done, info = env.step(action * cfg.max_force)
			time.sleep(cfg.dt)

			if done:
				print(f"Died! {info}")
				state = env.reset()

	except KeyboardInterrupt:
		print("Closing app...")
	finally:
		env.close()


if __name__ == "__main__":
	train(continue_train=True, save_log_process=True)
	# run()
