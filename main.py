import os
import random
import time

import numpy as np
import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.agent.memory import UniformReplayBuffer
from src.checkpointer.checkpointer import ModelCheckpointer
from src.config import Config
from src.physics.env import DoublePendulumEnv
from src.trainer.trainer import DDPGTrainer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# random.seed(23)
# np.random.seed(23)
# tf.random.set_seed(23)


def train(continue_train: bool = True, save_log_process: bool = True):
	cfg: Config = Config()

	env = DoublePendulumEnv(cfg, render=False)
	buffer = UniformReplayBuffer(cfg.buffer_maxsize)
	agent = DDPGAgent(cfg)
	trainer = DDPGTrainer(cfg, env, agent, buffer)
	checkpointer = ModelCheckpointer(agent)

	if continue_train:
		checkpointer.load_latest()
		start_episode: int = int(checkpointer.episode_counter) + 1
	else:
		start_episode = 1

	for episode in range(start_episode, start_episode + cfg.max_episodes):
		print(f"Running episode {episode:4}...")

		episode_reward, steps_survived, actor_loss, critic_loss, avg_q, difficulty = (
			trainer.run_episode()
		)

		if save_log_process and episode % cfg.log_every_n_episodes == 0:
			checkpointer.episode_counter.assign(episode)
			checkpointer.log_scalar(
				"Episode Total Reward", episode_reward, step=episode
			)
			checkpointer.log_scalar(
				"Steps Survived in Each Episode", steps_survived, step=episode
			)
			checkpointer.log_scalar(
				"Episode Average Actor Loss", actor_loss, step=episode
			)
			checkpointer.log_scalar(
				"Episode Average Critic Loss", critic_loss, step=episode
			)
			checkpointer.log_scalar("Episode Average Q-Values", avg_q, step=episode)
			checkpointer.log_scalar("Difficulty Level", difficulty.level, step=episode)

			checkpointer.save(episode)


def run():
	cfg: Config = Config()

	env = DoublePendulumEnv(cfg, render=True)
	env.set_difficulty(reset_angle_range_deg=180, angle_threshold_deg=181)

	agent = DDPGAgent(cfg)

	checkpointer = ModelCheckpointer(agent)
	checkpointer.load_latest()

	try:
		state = env.reset()
		while True:
			# Get the action from the actor
			action: float = agent.get_action(state, noise=0)

			# Step the environment based on the action
			state, *_ = env.step(action * cfg.max_force)
			
			time.sleep(1 / 240)

	except KeyboardInterrupt:
		env.close()


if __name__ == "__main__":
	train(continue_train=True, save_log_process=True)
	# run()
