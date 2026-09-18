import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.checkpointer.checkpointer import ModelCheckpointer
from src.config import PROJECT_ROOT, Config
from src.physics.env import DoublePendulumEnv


def run(checkpoint: Path | None = None) -> None:
	cfg: Config = Config()
	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	tf.random.set_seed(cfg.seed)

	env = DoublePendulumEnv(cfg, render=True)
	try:
		env.set_difficulty(reset_angle_range_deg=15, angle_threshold_deg=24)
		agent = DDPGAgent(cfg)
		checkpointer = ModelCheckpointer(agent)

		if checkpoint is not None:
			if not checkpoint.is_absolute():
				checkpoint = PROJECT_ROOT / checkpoint
			checkpointer.load_checkpoint(checkpoint)
		elif not checkpointer.load_latest():
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
