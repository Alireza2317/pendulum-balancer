import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.agent.memory import Batch, IBuffer, Transition
from src.config import Config
from src.physics.env import DoublePendulumEnv
from src.physics.state import EnvState
from src.trainer.curriculum import CurriculumManager, DifficultyParams
from src.trainer.evaluation import AgentMetrics, EpisodeResult
from src.trainer.explore import ExplorationScheduler
from src.trainer.noise import OUNoise


class ResetMode(Enum):
	UNIFORM = "uniform"
	OPPOSING_POSITIVE_FIRST = "opposing_positive_first"
	OPPOSING_NEGATIVE_FIRST = "opposing_negative_first"


@dataclass(frozen=True)
class ResetInfo:
	mode: ResetMode
	pole1_angle_deg: float
	pole2_angle_deg: float


class DDPGTrainer:
	def __init__(
		self,
		config: Config,
		environment: DoublePendulumEnv,
		agent: DDPGAgent,
		replay_buffer: IBuffer,
	) -> None:
		self.cfg = config
		self.env: DoublePendulumEnv = environment
		self.agent: DDPGAgent = agent
		self.noise: OUNoise = OUNoise(self.cfg)
		self.buffer: IBuffer = replay_buffer
		self.curriculum: CurriculumManager = CurriculumManager(self.cfg)
		self.exploration_scheduler = ExplorationScheduler(self.cfg, self.noise)
		self.last_reset_info: ResetInfo | None = None

	def _seed_episode(self, episode_number: int) -> None:
		"""Make an episode reproducible independently of process restarts."""

		episode_seed = self.cfg.seed + episode_number

		random.seed(episode_seed)
		np.random.seed(episode_seed)
		tf.random.set_seed(episode_seed)

	def _reset_mode(self, episode_number: int) -> ResetMode:
		cycle = self.cfg.targeted_reset_cycle
		targeted = self.cfg.targeted_reset_count

		if targeted == 0:
			return ResetMode.UNIFORM

		position = (episode_number - 1) % cycle
		spacing = cycle // targeted

		# Targeted episodes occur evenly throughout the cycle.
		if (position + 1) % spacing != 0:
			return ResetMode.UNIFORM

		targeted_index = (position + 1) // spacing - 1
		cycle_index = (episode_number - 1) // cycle
		global_targeted_index = cycle_index * targeted + targeted_index

		if global_targeted_index % 2 == 0:
			return ResetMode.OPPOSING_POSITIVE_FIRST

		return ResetMode.OPPOSING_NEGATIVE_FIRST

	def _generate_initial_state(
		self, difficulty: DifficultyParams, mode: ResetMode
	) -> EnvState:
		angle_range = difficulty.reset_angle_range_deg

		if mode == ResetMode.UNIFORM:
			pole1_deg = np.random.uniform(-angle_range, angle_range)
			pole2_deg = np.random.uniform(-angle_range, angle_range)
		else:
			magnitude1 = np.random.uniform(0.7 * angle_range, angle_range)
			magnitude2 = np.random.uniform(0.7 * angle_range, angle_range)

			if mode == ResetMode.OPPOSING_POSITIVE_FIRST:
				pole1_deg, pole2_deg = magnitude1, -magnitude2
			else:
				pole1_deg, pole2_deg = -magnitude1, magnitude2

		return EnvState(
			pole1_angle=float(np.deg2rad(pole1_deg)),
			pole2_angle=float(np.deg2rad(pole2_deg)),
			cart_x=0.0,
			cart_x_velocity=0.0,
			pole1_angular_velocity=0.0,
			pole2_angular_velocity=0.0,
		)

	def update_networks(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor] | None:
		if len(self.buffer) < self.cfg.buffer_warmup_size:
			return

		batch: Batch = self.buffer.sample(batch_size=self.cfg.batch_size)
		return self.agent.train_step(
			states=tf.convert_to_tensor(batch.states, dtype=tf.float32),
			actions=tf.expand_dims(
				tf.convert_to_tensor(batch.actions, dtype=tf.float32), axis=1
			),
			rewards=tf.expand_dims(
				tf.convert_to_tensor(batch.rewards, dtype=tf.float32), axis=1
			),
			next_states=tf.convert_to_tensor(batch.next_states, dtype=tf.float32),
			dones=tf.expand_dims(
				tf.convert_to_tensor(batch.dones, dtype=tf.float32), axis=1
			),
		)

	def _is_balanced(self, state: EnvState) -> bool:
		pole1_balanced = abs(state.pole1_angle) < np.deg2rad(
			self.cfg.balance_angle_threshold_deg
		)
		pole2_balanced = abs(state.pole2_angle) < np.deg2rad(
			self.cfg.balance_angle_threshold_deg
		)
		pole1_slow = (
			abs(state.pole1_angular_velocity) < self.cfg.balance_velocity_threshold
		)
		pole2_slow = (
			abs(state.pole2_angular_velocity) < self.cfg.balance_velocity_threshold
		)
		cart_centered = abs(state.cart_x) < self.cfg.balance_cart_threshold

		return (
			pole1_balanced
			and pole2_balanced
			and pole1_slow
			and pole2_slow
			and cart_centered
		)

	def evaluate_episode(self, initial_state: EnvState | None = None) -> EpisodeResult:
		state = self.env.reset(initial_state)

		total_reward: float = 0.0
		steps_performed: int = 0
		balanced_steps: int = 0
		failure_reason: str | None = None

		for _ in range(self.cfg.max_episode_steps):
			steps_performed += 1

			action = self.agent.get_action(state, noise=0.0)
			state, reward, done, info = self.env.step(action * self.cfg.max_force)

			total_reward += reward

			if self._is_balanced(state):
				balanced_steps += 1

			if done:
				failure_reason = info.get("reason")
				break

		return EpisodeResult(
			total_reward,
			steps_performed,
			balanced_steps / steps_performed,
			failure_reason,
		)

	def run_episode(
		self,
		episode_number: int | None = None,
	) -> tuple[EpisodeResult, AgentMetrics, DifficultyParams]:
		"""
		Run a loop until the action results in a state that is considered done.
		In each iteration of the loop:
			- Gets an action from the actor
			- Advances the environment
			- Saves the transition in the replay buffer
			- Updates all 4 networks parameters.

		Returns  a tuple containing:
			- An EpisodeResult object containing information of episode.
			- An AgentMetrics object containing information of the agent metrics.
			- Curriculum difficulty parameters.
		"""
		difficulty: DifficultyParams = self.curriculum.current_params()
		self.env.set_difficulty(
			reset_angle_range_deg=difficulty.reset_angle_range_deg,
		)

		self.exploration_scheduler.on_episode_start(
			difficulty.level,
			self.curriculum.success_ratio,
			self.curriculum.episodes_since_advance,
		)

		if episode_number is None:
			state = self.env.reset()
			reset_mode = ResetMode.UNIFORM
		else:
			self._seed_episode(episode_number)
			reset_mode = self._reset_mode(episode_number)
			initial_state = self._generate_initial_state(difficulty, reset_mode)
			state = self.env.reset(initial_state)

		self.last_reset_info = ResetInfo(
			mode=reset_mode,
			pole1_angle_deg=float(np.rad2deg(state.pole1_angle)),
			pole2_angle_deg=float(np.rad2deg(state.pole2_angle)),
		)

		done: bool = False
		failure_reason: str | None = None
		total_reward: float = 0

		actor_losses: list[float] = []
		critic_losses: list[float] = []
		episode_avg_q_vals: list[float] = []

		steps_balanced: int = 0
		steps_performed: int = 0
		for step in range(self.cfg.max_episode_steps):
			if done:
				break
			# Get the action from the actor
			action: float = self.agent.get_action(state, noise=self.noise.sample())

			# Step the environment based on the action
			next_state, reward, done, info = self.env.step(action * self.cfg.max_force)

			total_reward += reward
			failure_reason = info.get("reason")

			# Check balance
			if self._is_balanced(next_state):
				steps_balanced += 1

			# Save the transition into the buffer
			self.buffer.add(Transition(state, action, reward, next_state, done))

			state = next_state
			if step % self.cfg.train_every_n_steps == 0:
				result: tuple[tf.Tensor, tf.Tensor, tf.Tensor] | None = (
					self.update_networks()
				)
				if result is not None:
					actor_loss, critic_loss, q_vals = result
					actor_losses.append(float(actor_loss))
					critic_losses.append(float(critic_loss))
					episode_avg_q_vals.append(float(tf.reduce_mean(q_vals)))

			steps_performed += 1

		balance_fraction: float = steps_balanced / steps_performed

		self.curriculum.record_episode(steps_performed, balance_fraction, failed=done)

		episode_result = EpisodeResult(
			total_reward,
			steps_performed,
			balance_fraction,
			failure_reason,
		)
		agent_metrics = AgentMetrics(
			actor_loss=sum(actor_losses) / len(actor_losses) if actor_losses else 0.0,
			critic_loss=(
				sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
			),
			q_vals_avg=(
				sum(episode_avg_q_vals) / len(episode_avg_q_vals)
				if episode_avg_q_vals
				else 0.0
			),
		)

		return episode_result, agent_metrics, difficulty
