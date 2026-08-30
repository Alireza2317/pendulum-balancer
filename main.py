import os
import random

import numpy as np
import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.agent.memory import UniformReplayBuffer
from src.checkpointer.checkpointer import ModelCheckpointer
from src.physics.env import DoublePendulumEnv
from src.trainer.trainer import DDPGTrainer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
random.seed(23)
np.random.seed(23)
tf.random.set_seed(23)


def main():
	env = DoublePendulumEnv(render=True)
	buffer = UniformReplayBuffer(10_000)
	agent = DDPGAgent()
	trainer = DDPGTrainer(env, agent, buffer, batch_size=32)
	checkpointer = ModelCheckpointer(agent)

	checkpointer.load_latest()

	start_episode: int = int(checkpointer.episode_counter) + 1
	MAX_EPISODES: int = 1_000

	for episode in range(start_episode, start_episode + MAX_EPISODES):
		print(f"Running episode {episode:4}...")

		episode_reward = trainer.run_episode()

		if episode % 20 == 0:
			checkpointer.episode_counter.assign(episode)
			checkpointer.log_scalar(
				"Episode Total Reward", episode_reward, step=episode
			)
			checkpointer.save(episode)


if __name__ == "__main__":
	main()
