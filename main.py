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
from src.trainer.trainer import DDPGTrainer

# random.seed(23)
# np.random.seed(23)
# tf.random.set_seed(23)


def train(continue_train: bool = True, save_log_process: bool = True):
	cfg: Config = Config()

	env = DoublePendulumEnv(cfg, render=False)
	try:
		buffer = UniformReplayBuffer(cfg.buffer_maxsize)
		agent = DDPGAgent(cfg)
		trainer = DDPGTrainer(cfg, env, agent, buffer)
		checkpointer = ModelCheckpointer(agent, trainer)

		if continue_train:
			# checkpointer.load_latest()
			checkpointer.load_checkpoint(
				PROJECT_ROOT / "checkpoints" / "ckpt-1980", load_sidefiles=False
			)
			trainer.buffer.load(PROJECT_ROOT / "checkpoints" / "buffer_2000.pkl")
			trainer.curriculum.set_level(0)
			# start_episode: int = int(checkpointer.episode_counter) + 1
			start_episode: int = 2001
		else:
			start_episode = 1

		for episode in range(start_episode, start_episode + cfg.max_episodes):
			print(f"Running episode {episode:4}...")

			episode_result, agent_metric, difficulty = trainer.run_episode()

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
	run()
