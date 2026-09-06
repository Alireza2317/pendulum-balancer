from typing import Any

import numpy as np
import pybullet as p
import pybullet_data

from src.config import Config
from src.physics.angles import abs2rel, denormalize_angle, normalize_angle, rel2abs
from src.physics.state import EnvState


class DoublePendulumEnv:
	def __init__(self, config: Config, render: bool = True) -> None:
		connection_mode = p.GUI if render else p.DIRECT
		self.client_id: int = p.connect(connection_mode)
		self.cfg = config

		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, self.cfg.gravity, physicsClientId=self.client_id)

		self.cart_id = p.loadURDF(
			self.cfg.pendulum_urdf_path,
			useFixedBase=True,
			physicsClientId=self.client_id,
		)
		self._disable_motors()

		# Curriculum-controlled difficulty, defaults to the easiest setting.
		self._reset_angle_range_deg: float = self.cfg.curriculum_reset_start_deg
		self._angle_threshold_deg: float = (
			self.cfg.curriculum_reset_start_deg + self.cfg.curriculum_margin_start_deg
		)

		self.reset()

	def set_difficulty(
		self, reset_angle_range_deg: float, angle_threshold_deg: float
	) -> None:
		"""
		Sets how far poles are randomized at reset, and the angle (deg) that
		fails the episode.
		"""
		self._reset_angle_range_deg = reset_angle_range_deg
		self._angle_threshold_deg = angle_threshold_deg

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
				jointDamping=self.cfg.joint_damping,
				physicsClientId=self.client_id,
			)

	def get_state(self) -> EnvState:
		cart_info = p.getJointState(self.cart_id, 0, physicsClientId=self.client_id)
		pole1_info = p.getJointState(self.cart_id, 1, physicsClientId=self.client_id)
		pole2_info = p.getJointState(self.cart_id, 2, physicsClientId=self.client_id)

		angle1: float = pole1_info[0]
		angle2_rel: float = pole2_info[0]
		angle2_abs: float = rel2abs(angle1, angle2_rel)

		state: EnvState = EnvState(
			cart_x=cart_info[0],
			cart_x_velocity=cart_info[1],
			pole1_angle=normalize_angle(angle1),
			pole1_angular_velocity=float(
				np.clip(pole1_info[1], -self.cfg.max_velocity, self.cfg.max_velocity)
			),
			pole2_angle=normalize_angle(angle2_abs),
			pole2_angular_velocity=float(
				np.clip(pole2_info[1], -self.cfg.max_velocity, self.cfg.max_velocity)
			),
		)

		return state

	def _set_state(self, state: EnvState) -> None:
		"""
		Takes in a EnvState object and applies it to the environment.
		It also converts the absolute angle form to the relative form which is needed
		by pybullet's system.
		"""
		# Cart
		p.resetJointState(
			self.cart_id,
			jointIndex=0,  # The cart's index itself, inside the urdf file
			targetValue=state.cart_x,
			targetVelocity=state.cart_x_velocity,
			physicsClientId=self.client_id,
		)

		angle1_original: float = denormalize_angle(state.pole1_angle)
		# Pole 1
		p.resetJointState(
			self.cart_id,
			jointIndex=1,  # Pole 1 index
			targetValue=angle1_original,
			targetVelocity=state.pole1_angular_velocity,
			physicsClientId=self.client_id,
		)

		angle2_original: float = denormalize_angle(state.pole2_angle)
		pole2_angle_relative: float = abs2rel(angle1_original, angle2_original)
		# Pole 2
		p.resetJointState(
			self.cart_id,
			jointIndex=2,  # Pole 2 index
			targetValue=pole2_angle_relative,
			targetVelocity=state.pole2_angular_velocity,
			physicsClientId=self.client_id,
		)

	def reset(self) -> EnvState:
		"""Resets the environment randomly and returns the initial state."""
		DELTA_DEG: float = self._reset_angle_range_deg

		angle1_random_deg = np.random.uniform(-DELTA_DEG, +DELTA_DEG)
		angle2_random_deg = np.random.uniform(-DELTA_DEG, DELTA_DEG)

		random_state: EnvState = EnvState(
			cart_x=0,
			cart_x_velocity=0,
			pole1_angle=np.deg2rad(angle1_random_deg),
			pole2_angle=np.deg2rad(angle2_random_deg),
			pole1_angular_velocity=0,
			pole2_angular_velocity=0,
		)

		self._set_state(random_state)
		self._prev_cost: float = self._cost(random_state)

		return self.get_state()

	def _cost(self, state: EnvState) -> float:
		"""Combined, bounded cost which is 0 for perfectly upright and centered cart"""
		# Bounded upright-ness cost per pole: 0 when upright, 2 when hanging.
		# Penalize large angles and deviations from the upright position
		angle1_cost: float = 1 - np.cos(state.pole1_angle)
		angle2_cost: float = 1 - np.cos(state.pole2_angle)

		# Penalize cart getting farther from the origin(x=0)
		cart_x_cost: float = state.cart_x**2

		return angle1_cost + angle2_cost + cart_x_cost

	def _calculate_reward(self, state: EnvState) -> float:
		"""
		Reward is kept strictly positive: alive_bonus is set above the worst possible
		combined cost, so ending the episode to escape negative accumulation is worse
		than positive future reward. A potential-based progress term additionally
		rewards genuinely reducing the cost step-to-step, not just its absolute size.
		"""
		cost: float = self._cost(state)

		# Reward for reducing cost since the previous step.
		# (positive if improving, negative if getting worse).
		# Clipped defensively, so round-trips (worsen then recover to the same angle)
		# nets to ~0, avoiding farming exploit from oscillating in place.
		cost_delta: float = np.clip(
			self._prev_cost - cost,
			-self.cfg.progress_cost_clip,
			self.cfg.progress_cost_clip,
		)

		self._prev_cost = cost

		progress_reward: float = self.cfg.progress_reward_scale * cost_delta

		return (
			self.cfg.alive_bonus
			+ progress_reward
			- cost
			- (self.cfg.terminal_penalty if self._is_hard_fail(state) else 0)
		)

	def _is_hard_fail(self, state: EnvState) -> bool:
		"""True only for the unrecoverable failure: cart off the rail."""
		return abs(state.cart_x) > self.cfg.cart_x_threshold

	def _is_done(self, state: EnvState) -> bool:
		"""
		The episode is finished if either of these conditions are met:
			1. The cart is off the rail (real, unrecoverable failure).
			2. A pole angle exceeds the curriculum's current threshold.
		At low curriculum levels (2) is tight, so the agent learns fine balance without
		wasting steps recovering from a fall. At high curriculum levels the threshold is
		relaxed past 180 deg so it can never trigger, and only (1) or the maximum
		episode's step cap end things; forcing the agent to recover from falls.
		"""
		angle_threshold: float = np.deg2rad(self._angle_threshold_deg)

		return (
			self._is_hard_fail(state)
			or abs(state.pole1_angle) > angle_threshold
			or abs(state.pole2_angle) > angle_threshold
		)

	def step(self, action: float) -> tuple[EnvState, float, bool, dict[str, Any]]:
		"""
		Applies an action(the force), steps physics,
		and returns (next state, reward, done, info).
		"""
		force: float = float(np.clip(action, -self.cfg.max_force, self.cfg.max_force))

		p.setJointMotorControl2(
			self.cart_id,
			0,
			controlMode=p.TORQUE_CONTROL,  # Slide motion
			force=force,
			physicsClientId=self.client_id,
		)

		p.stepSimulation(physicsClientId=self.client_id)

		new_state: EnvState = self.get_state()
		reward: float = self._calculate_reward(new_state)
		done: bool = self._is_done(new_state)
		return new_state, reward, done, {}

	def close(self) -> None:
		p.disconnect(physicsClientId=self.client_id)
