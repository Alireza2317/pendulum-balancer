import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.agent.memory import Batch, IBuffer, Transition
from src.config import Config
from src.physics.env import DoublePendulumEnv
from src.trainer.curriculum import CurriculumManager, DifficultyParams
from src.trainer.noise import OUNoise


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

		self.exploration_decay_active: bool = False

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

	def run_episode(self) -> tuple[float, float, float, float, float, DifficultyParams]:
		"""
		Run a loop until the action results in a state that is considered done.
		In each iteration of the loop:
			- Gets an action from the actor
			- Advances the environment
			- Saves the transition in the replay buffer
			- Updates all 4 networks parameters.

		Returns  a tuple containing:
			- The total reward earned in the episode.
			- Number of steps survived in the episode.
			- Average actor loss.
			- Average critic loss.
			- Average Q-Values.
			- Curriculum difficulty parameters.
		"""
		difficulty: DifficultyParams = self.curriculum.current_params()
		self.env.set_difficulty(
			reset_angle_range_deg=difficulty.reset_angle_range_deg,
			angle_threshold_deg=difficulty.angle_threshold_deg,
		)
		state = self.env.reset()

		self.noise.reset()
		if self.exploration_decay_active:
			self.noise.decay()

		done: bool = False
		total_reward: float = 0

		actor_losses: list[float] = []
		critic_losses: list[float] = []
		episode_avg_q_vals: list[float] = []

		step: int = 0
		for step in range(self.cfg.max_episode_steps):
			if done:
				break
			# Get the action from the actor
			action: float = self.agent.get_action(state, noise=self.noise.sample())

			# Step the environment based on the action
			next_state, reward, done, _ = self.env.step(action * self.cfg.max_force)

			total_reward += reward

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

		steps_survived: int = step + 1
		self.curriculum.record_episode(steps_survived)
		if self.curriculum.success_ratio >= self.cfg.exploration_decay_unlock_threshold:
			self.exploration_decay_active = True

		avg_actor_loss: float = (
			sum(actor_losses) / len(actor_losses) if actor_losses else 0.0
		)
		avg_critic_loss: float = (
			sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
		)
		avg_q: float = (
			sum(episode_avg_q_vals) / len(episode_avg_q_vals)
			if episode_avg_q_vals
			else 0.0
		)
		return (
			total_reward,
			steps_survived,
			avg_actor_loss,
			avg_critic_loss,
			avg_q,
			difficulty,
		)
