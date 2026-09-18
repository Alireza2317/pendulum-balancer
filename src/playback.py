import random
import time
from pathlib import Path

import numpy as np
import pybullet as p
import tensorflow as tf

from src.agent.agent import TD3Agent
from src.checkpointer.checkpointer import ModelCheckpointer
from src.config import PROJECT_ROOT, Config
from src.physics.env import DoublePendulumEnv


def add_reset_button(env: DoublePendulumEnv) -> tuple[int, float]:
	reset_button = p.addUserDebugParameter(
		"Reset environment (R)",
		1,
		0,
		0,
		physicsClientId=env.client_id,
	)
	button_value = p.readUserDebugParameter(
		reset_button,
		physicsClientId=env.client_id,
	)
	return reset_button, button_value


def run_simulation(
	env: DoublePendulumEnv,
	agent: TD3Agent,
	cfg: Config,
	reset_button: int,
	reset_button_value: float,
) -> None:
	state = env.reset()
	while True:
		keys = p.getKeyboardEvents(physicsClientId=env.client_id)
		new_reset_button_value = p.readUserDebugParameter(
			reset_button,
			physicsClientId=env.client_id,
		)
		reset_requested = (
			keys.get(ord("r"), 0) & p.KEY_WAS_TRIGGERED
			or new_reset_button_value != reset_button_value
		)
		reset_button_value = new_reset_button_value

		if reset_requested:
			state = env.reset()
			print("Environment reset.")
			continue

		action: float = agent.get_action(state, noise=0)
		state, _, done, info = env.step(action * cfg.max_force)
		time.sleep(cfg.dt)

		if done:
			print(f"Died! {info}")
			state = env.reset()


def run(checkpoint: Path | None = None) -> None:
	cfg: Config = Config()
	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	tf.random.set_seed(cfg.seed)

	env = DoublePendulumEnv(cfg, render=True)
	try:
		env.set_difficulty(reset_angle_range_deg=15)
		agent = TD3Agent(cfg)
		checkpointer = ModelCheckpointer(agent)

		if checkpoint is not None:
			if not checkpoint.is_absolute():
				checkpoint = PROJECT_ROOT / checkpoint
			checkpointer.load_checkpoint(checkpoint)
		elif not checkpointer.load_latest():
			raise FileNotFoundError("No checkpoints found!")

		print("Checkpoint loaded successfully!")
		print("Press R or use the on-screen button to reset the environment.")
		reset_button, reset_button_value = add_reset_button(env)
		run_simulation(env, agent, cfg, reset_button, reset_button_value)

	except KeyboardInterrupt:
		print("Closing app...")
	finally:
		env.close()
