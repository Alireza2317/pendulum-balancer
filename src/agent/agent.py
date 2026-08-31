from dataclasses import dataclass

import tensorflow as tf
import tensorflow.keras as tfk

from src.agent.models import Actor, Critic
from src.config import Config
from src.physics.state import EnvState


class DDPGAgent:
	def __init__(self, cfg: Config) -> None:
		self.cfg = cfg

		self.actor = Actor()
		self.critic = Critic()
		self.target_actor = Actor()
		self.target_critic = Critic()

		# Build initial weights with dummy inputs
		dummy_state = tf.zeros((1, 6))
		dummy_action = tf.zeros((1, 1))
		self.actor(dummy_state)
		self.target_actor(dummy_state)
		self.critic(dummy_state, dummy_action)
		self.target_critic(dummy_state, dummy_action)

		self.target_actor.set_weights(self.actor.get_weights())
		self.target_critic.set_weights(self.critic.get_weights())

		self.actor_optimizer = tfk.optimizers.Adam(
			learning_rate=self.cfg.actor_lr, clipnorm=1
		)
		self.critic_optimizer = tfk.optimizers.Adam(
			learning_rate=self.cfg.critic_lr, clipnorm=1
		)

	@tf.function
	def update_target_networks(self) -> None:
		"""Applies Polyak averaging (theta' = tau * theta + (1-tau) * theta')."""

		for target, main in zip(self.target_actor.variables, self.actor.variables):
			target.assign(target * (1 - self.cfg.tau) + main * self.cfg.tau)

		for target, main in zip(self.target_critic.variables, self.critic.variables):
			target.assign(target * (1 - self.cfg.tau) + main * self.cfg.tau)

	def get_action(self, state: EnvState, noise_std: float) -> float:
		"""Runs inference and adds exploration noise during training."""
		state_tensor: tf.Tensor = tf.convert_to_tensor(
			state.nparray(), dtype=tf.float32
		)
		# Expand state's dimension from (n, ) to (1, n)
		state_tensor = tf.expand_dims(state_tensor, axis=0)

		action: tf.Tensor = self.actor(state_tensor)[0, 0]

		if noise_std > 0:
			noise = tf.random.normal(shape=(), mean=0, stddev=noise_std)
			action += noise

		return float(tf.clip_by_value(action, -1, 1))

	@tf.function
	def train_step(
		self,
		states: tf.Tensor,
		actions: tf.Tensor,
		rewards: tf.Tensor,
		next_states: tf.Tensor,
		dones: tf.Tensor,
	) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
		"""
		Calculates loss and applies gradients for one sampled batch.
		Returns actor's loss and critic's loss and predicted Q-values
		"""
		# Update Critic
		with tf.GradientTape() as tape:
			# The critic should have outputs close to target values y
			# y = r + gamma * Q(s', mu'(s'))
			# Calculate the target value
			next_actions: tf.Tensor = self.target_actor(next_states)
			# Future rewards
			target_q_values: tf.Tensor = self.target_critic(next_states, next_actions)
			# If the episode was ended on a certain transition
			# There are no future rewards, so we need the (1-done) coefficient
			future_rewards: tf.Tensor = target_q_values * (1 - dones)
			y: tf.Tensor = tf.stop_gradient(rewards + self.cfg.gamma * future_rewards)

			# MSE loss
			predicted_q_values: tf.Tensor = self.critic(states, actions)
			critic_loss: tf.Tensor = tf.reduce_mean(tf.square(y - predicted_q_values))

		critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
		self.critic_optimizer.apply_gradients(
			zip(critic_grads, self.critic.trainable_variables)
		)

		# Update Actor
		with tf.GradientTape() as tape:
			# The actor should maximize the critic's reward predictions Q(s, mu(s))
			actor_actions: tf.Tensor = self.actor(states)

			# Since gradients try to MINIMIZE, the actor should minimize -Q(s, mu(s))
			actor_loss: tf.Tensor = -tf.reduce_mean(self.critic(states, actor_actions))

		actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
		self.actor_optimizer.apply_gradients(
			zip(actor_grads, self.actor.trainable_variables)
		)

		# Update the target networks, very slightly and softly
		self.update_target_networks()

		return actor_loss, critic_loss, predicted_q_values
