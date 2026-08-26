from typing import Any

import pybullet as p
import pybullet_data

from src.physics.state import EnvState


class DoublePendulumEnv:
	def __init__(self, render: bool = True) -> None:
		connection_mode = p.GUI if render else p.DIRECT
		self.client_id: int = p.connect(connection_mode)

		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)

		self.cart_id = p.loadURDF(
			"assets/urdf/pendulum.urdf",
			useFixedBase=True,
			physicsClientId=self.client_id,
		)
		self._disable_motors()

	def _disable_motors(self) -> None:
		"""Frees joints so physics (gravity/inertia) drives them."""
		for i in range(p.getNumJoints(self.cart_id, physicsClientId=self.client_id)):
			# Disable default motors
			p.setJointMotorControl2(
				self.cart_id,
				i,
				controlMode=p.VELOCITY_CONTROL,
				force=0,
				physicsClientId=self.client_id,
			)

			# Add small amount of friction for realism
			p.changeDynamics(
				self.cart_id,
				i,
				jointDamping=0.05,
				physicsClientId=self.client_id
			)

	def get_state(self) -> EnvState:
		cart_info = p.getJointState(self.cart_id, 0, physicsClientId=self.client_id)
		pole1_info = p.getJointState(self.cart_id, 1, physicsClientId=self.client_id)
		pole2_info = p.getJointState(self.cart_id, 2, physicsClientId=self.client_id)

		return EnvState(
			cart_x=cart_info[0],
			cart_x_velocity=cart_info[1],
			pole1_angle=pole1_info[0],
			pole1_angular_velocity=pole1_info[1],
			pole2_angle=pole2_info[0],
			pole2_angular_velocity=pole2_info[1],
		)

	def reset(self) -> EnvState:
		"""Resets the environment and returns the initial state."""
		# Set cart's x and x velocity to 0
		p.resetJointState(
			self.cart_id,
			jointIndex=0,  # The cart's index itself, inside the urdf file
			targetValue=0,
			targetVelocity=0,
			physicsClientId=self.client_id,
		)

		for joint_index in (1, 2):
			angle_noise = 0
			# angle_noise = np.random.uniform(-0.05, 0.05)
			p.resetJointState(
				self.cart_id,
				jointIndex=joint_index,
				targetValue=angle_noise,
				targetVelocity=0,
				physicsClientId=self.client_id,
			)

		return self.get_state()

	def step(self, action: float) -> tuple[EnvState, float, bool, dict[str, Any]]:
		"""Applies an action, steps physics, and returns (next state, reward, done, info)."""
		# 1. Apply force (action) to the cart's prismatic joint
		# 2. p.stepSimulation(physicsClientId=self.client_id)
		# 3. Read new joint states to form the observation array
		# 4. Calculate reward and check if terminal
		p.setJointMotorControl2(
			self.cart_id,
			0,
			controlMode=p.TORQUE_CONTROL,
			force=action,
			physicsClientId=self.client_id,
		)

		p.stepSimulation(physicsClientId=self.client_id)

		return self.get_state(), 0, False, {}

	def close(self) -> None:
		p.disconnect(physicsClientId=self.client_id)
