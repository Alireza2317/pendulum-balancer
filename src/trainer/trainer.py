import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.agent.memory import Batch, IBuffer, Transition
from src.physics.env import DoublePendulumEnv
from src.physics.state import EnvState


class DDPGTrainer:
	def __init__(
		self,
		environment: DoublePendulumEnv,
		agent: DDPGAgent,
		replay_buffer: IBuffer,
		batch_size: int = 64,
	) -> None:
		self.env: DoublePendulumEnv = environment
		self.agent: DDPGAgent = agent
		self.buffer: IBuffer = replay_buffer
		self.BS: int = batch_size

	def collect_transition(self, state: EnvState) -> tuple[EnvState, bool]:
		"""
		- Gets an action from the actor
		- Advances the environment
		- Saves the transition in the replay buffer
		Returns the next state and is_done
		"""
		# Get the action from the actor
		state_tensor: tf.Tensor = tf.convert_to_tensor(
			state.nparray(), dtype=tf.float32
		)
		action: float = self.agent.get_action(state_tensor, noise_std=0.1)

		# Step the environment based on the action
		next_state, reward, done, info = self.env.step(action)

		# Save the transition into the buffer
		self.buffer.add(
			Transition(
				state=state,
				action=action,
				reward=reward,
				next_state=next_state,
				done=done,
			)
		)
		return next_state, done

	def update_networks(self) -> None:
		if len(self.buffer) < self.BS:
			return

		batch: Batch = self.buffer.sample(batch_size=self.BS)
		self.agent.train_step(
			states=tf.convert_to_tensor(batch.states, dtype=tf.float32),
			actions=tf.expand_dims(tf.convert_to_tensor(batch.actions, dtype=tf.float32), axis=1),
			rewards=tf.expand_dims(tf.convert_to_tensor(batch.rewards, dtype=tf.float32), axis=1),
			next_states=tf.convert_to_tensor(batch.next_states, dtype=tf.float32),
			dones=tf.expand_dims(tf.convert_to_tensor(batch.dones, dtype=tf.float32), axis=1),
		)

	def run_episode(self) -> None:
		state = self.env.reset()
		done: bool = False

		while not done:
			state, done = self.collect_transition(state)
			self.update_networks()
