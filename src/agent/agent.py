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
		self.critic2 = Critic()
		self.target_actor = Actor()
		self.target_critic = Critic()
		self.target_critic2 = Critic()

		# Build initial weights with dummy inputs
		self._build_weights()

		self._sync_target_networks()

		self.actor_optimizer = tfk.optimizers.Adam(
			learning_rate=self.cfg.actor_lr, clipnorm=1
		)
		self.critic_optimizer = tfk.optimizers.Adam(
			learning_rate=self.cfg.critic_lr, clipnorm=1
		)

		self.actor_optimizer.build(self.actor.trainable_variables)

		critic_variables = (
			self.critic.trainable_variables + self.critic2.trainable_variables
		)
		self.critic_optimizer.build(critic_variables)

		# Tracks critic-update count, to gate delayed actor/target updates.
		self._step_counter = tf.Variable(
			tf.constant(0, dtype=tf.int64), trainable=False
		)
		# Reused for logging on steps where the actor doesn't update, so
		# logged actor loss stays meaningful instead of dropping to a
		# stale/zero value on skipped steps.
		self._last_actor_loss = tf.Variable(
			tf.constant(0.0, dtype=tf.float32), trainable=False
		)

	def _build_weights(self) -> None:
		dummy_state = tf.zeros((1, self.cfg.state_dim))
		dummy_action = tf.zeros((1, self.cfg.action_dim))
		self.actor(dummy_state)
		self.target_actor(dummy_state)
		self.critic(dummy_state, dummy_action)
		self.critic2(dummy_state, dummy_action)
		self.target_critic(dummy_state, dummy_action)
		self.target_critic2(dummy_state, dummy_action)

	def _sync_target_networks(self) -> None:
		self.target_actor.set_weights(self.actor.get_weights())
		self.target_critic.set_weights(self.critic.get_weights())
		self.target_critic2.set_weights(self.critic2.get_weights())

	def update_target_networks(self) -> None:
		"""Applies Polyak averaging (theta' = tau * theta + (1-tau) * theta')."""
		network_pairs: list[tuple[tfk.Model, tfk.Model]] = [
			(self.target_actor, self.actor),
			(self.target_critic, self.critic),
			(self.target_critic2, self.critic2),
		]

		for target_network, main_network in network_pairs:
			for target, main in zip(target_network.variables, main_network.variables):
				target.assign(target * (1 - self.cfg.tau) + main * self.cfg.tau)

	@tf.function
	def _fast_inference(self, state_tensor: tf.Tensor) -> tf.Tensor:
		return self.actor(state_tensor)[0, 0]

	def get_action(self, state: EnvState, noise: float) -> float:
		"""Runs inference and adds exploration noise during training."""
		state_tensor: tf.Tensor = tf.convert_to_tensor(
			state.nparray(), dtype=tf.float32
		)
		# Expand state's dimension from (n, ) to (1, n)
		state_tensor = tf.expand_dims(state_tensor, axis=0)

		action: tf.Tensor = self._fast_inference(state_tensor) + noise

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
		Returns a tuple of:
			- actor's loss (stale on delayed steps)
			- critic's loss
			- predicted Q-values
		"""
		# Update Critics

		# Target policy smoothing: perturb the target action slightly so the
		# critics are forced to be accurate over a neighborhood of actions,
		# not just one exact point the actor could otherwise exploit.
		smoothing_noise: tf.Tensor = tf.clip_by_value(
			tf.random.normal(tf.shape(actions), stddev=self.cfg.policy_noise),
			-self.cfg.noise_clip,
			self.cfg.noise_clip,
		)

		next_actions: tf.Tensor = tf.clip_by_value(
			self.target_actor(next_states) + smoothing_noise, -1, 1
		)

		# Twin critics: take the minimum of both target estimates. Counters
		# single-critic overestimation, since both networks would have to be
		# optimistic about the same action for the bias to persist.

		# Future rewards
		target_q1: tf.Tensor = self.target_critic(next_states, next_actions)
		target_q2: tf.Tensor = self.target_critic2(next_states, next_actions)
		target_q_values: tf.Tensor = tf.minimum(target_q1, target_q2)

		# If the episode was ended on a certain transition
		# There are no future rewards, so we need the (1-done) coefficient
		future_rewards: tf.Tensor = target_q_values * (1 - dones)

		# The critics should have outputs close to target values y
		# y = r + gamma * Q(s', mu'(s'))
		# Calculate the target value
		y: tf.Tensor = tf.stop_gradient(rewards + self.cfg.gamma * future_rewards)

		with tf.GradientTape() as tape:
			# MSE loss
			predicted_q_values: tf.Tensor = self.critic(states, actions)
			predicted_q_values2: tf.Tensor = self.critic2(states, actions)

			critic_loss: tf.Tensor = tf.reduce_mean(
				tf.square(y - predicted_q_values)
			) + tf.reduce_mean(tf.square(y - predicted_q_values2))

		critic_vars = self.critic.trainable_variables + self.critic2.trainable_variables
		critic_grads = tape.gradient(critic_loss, critic_vars)
		self.critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

		self._step_counter.assign_add(1)

		# Update Actor
		# Delayed actor + target network updates
		# Only every `policy_delay` critic updates: gives the critic time to
		# stabilize before the actor climbs it, and keeps target networks in
		# sync with that same slower cadence (standard TD3 pairing).
		if int(self._step_counter) % self.cfg.policy_delay == 0:
			with tf.GradientTape() as tape:
				# The actor should maximize the critic's reward predictions Q(s, mu(s))
				actor_actions: tf.Tensor = self.actor(states)

				# Since gradients try to MINIMIZE, the actor should minimize -Q(s, mu(s))
				actor_loss: tf.Tensor = -tf.reduce_mean(
					self.critic(states, actor_actions)
				)

			actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
			self.actor_optimizer.apply_gradients(
				zip(actor_grads, self.actor.trainable_variables)
			)

			# Update the target networks, very slightly and softly
			self.update_target_networks()

			self._last_actor_loss.assign(actor_loss)

		return self._last_actor_loss, critic_loss, predicted_q_values
