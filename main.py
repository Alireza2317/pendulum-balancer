import time

from src.physics.env import DoublePendulumEnv

env = DoublePendulumEnv()

for i in range(10000):
	env.step(0)
	time.sleep(1 / 240)

env.close()

