from typing import Any

import numpy as np
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
				self.cart_id, i, jointDamping=0.05, physicsClientId=self.client_id
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

	def _calculate_reward(self, state: EnvState, action: float) -> float:
		"""Calculates penalty based on angles, position, and action/force."""
		# Penalize large angles and deviations from the upright position
		angle1_cost: float = state.pole1_angle**2
		angle2_cost: float = state.pole2_angle**2

		# Penalize cart getting farther from the origin(x=0)
		cart_x_cost: float = 0.1 * (state.cart_x**2)

		# Penalize large forces
		action_cost: float = 0.01 * (action**2)

		return -(angle1_cost + angle2_cost + cart_x_cost + action_cost)

	def _is_done(self, state: EnvState) -> bool:
		"""
		The episode is finished if either of these conditions are met:
			1. If the cart is so close to the edges (<-0.95 or >0.95)
			2. If the poles angles exceed a certain threshold (e.g. 30 degrees)
		Returns True if the episode is finished.
		"""
		position_threshold: float = 0.95
		angle_threshold_deg: float = 30.0
		angle_threshold: float = np.deg2rad(angle_threshold_deg)

		return (
			abs(state.cart_x) > position_threshold
			or abs(state.pole1_angle) > angle_threshold
			or abs(state.pole2_angle) > angle_threshold
		)

	def step(self, action: float) -> tuple[EnvState, float, bool, dict[str, Any]]:
		"""
		Applies an action(the force), steps physics,
		and returns (next state, reward, done, info).
		"""
		# 1. Apply force (action) to the cart's prismatic joint
		# 2. p.stepSimulation(physicsClientId=self.client_id)
		# 3. Read new joint states to form the observation array
		# 4. Calculate reward and check if terminal
		force: float = float(np.clip(action, -100, 100))

		p.setJointMotorControl2(
			self.cart_id,
			0,
			controlMode=p.TORQUE_CONTROL,  # Slide motion
			force=force,
			physicsClientId=self.client_id,
		)

		p.stepSimulation(physicsClientId=self.client_id)

		new_state: EnvState = self.get_state()
		reward: float = self._calculate_reward(new_state, force)
		done: bool = self._is_done(new_state)
		return new_state, reward, done, {}

	def close(self) -> None:
		p.disconnect(physicsClientId=self.client_id)
