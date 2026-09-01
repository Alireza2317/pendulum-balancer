import time

import pybullet as p

from src.config import Config
from src.physics.env import DoublePendulumEnv

if __name__ == "__main__":
	cfg=Config()
	env = DoublePendulumEnv(cfg)
	force_slider_id = p.addUserDebugParameter(
		"Force", -10, 10, 0, physicsClientId=env.client_id
	)

	try:
		while True:
			force: float = p.readUserDebugParameter(
				force_slider_id, physicsClientId=env.client_id
			)
			env.step(force)
			time.sleep(1 / 240)
	except KeyboardInterrupt:
		env.close()
