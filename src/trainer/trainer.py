import tensorflow as tf

from src.agent.agent import DDPGAgent
from src.agent.memory import Batch, IBuffer, Transition
from src.physics.env import MAX_FORCE, DoublePendulumEnv


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

	def update_networks(self) -> None:
		if len(self.buffer) < 5_000:
			return

		batch: Batch = self.buffer.sample(batch_size=self.BS)
		self.agent.train_step(
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

	def run_episode(self, action_noise_std: float = 0.1) -> float:
		"""
		Run a loop until the action results in a state that is considered done.
		In each iteration of the loop:
			- Gets an action from the actor
			- Advances the environment
			- Saves the transition in the replay buffer
			- Updates all 4 networks parameters.

		Returns the total reward earned in the episode.
		"""

		state = self.env.reset()
		done: bool = False
		total_reward: float = 0

		while not done:
			# Get the action from the actor
			action: float = self.agent.get_action(state, noise_std=action_noise_std)

			# Step the environment based on the action
			next_state, reward, done, _ = self.env.step(action * MAX_FORCE)

			total_reward += reward

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

			self.update_networks()

		return total_reward
