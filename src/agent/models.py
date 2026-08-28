import tensorflow as tf
import tensorflow.keras as tfk


class Actor(tfk.Model):
	def __init__(self) -> None:
		super().__init__()
		self.dense1 = tfk.layers.Dense(256, activation="relu")
		self.dense2 = tfk.layers.Dense(256, activation="relu")
		self.out = tfk.layers.Dense(1, activation="tanh")

	def call(self, state: tf.Tensor) -> tf.Tensor:
		x = self.dense1(state)
		x = self.dense2(x)
		return self.out(x)


class Critic(tfk.Model):
	def __init__(self) -> None:
		super().__init__()

		# State path
		self.state_dense = tfk.layers.Dense(32, activation="relu")

		# Action path
		self.action_dense = tfk.layers.Dense(32, activation="relu")

		self.cat = tfk.layers.Concatenate()
		self.dense1 = tfk.layers.Dense(256, activation="relu")
		self.dense2 = tfk.layers.Dense(256, activation="relu")

		# Since Q-values are real numbers, they don't need any activation
		self.out = tfk.layers.Dense(1, activation=None)

	def call(self, state: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
		s = self.state_dense(state)
		a = self.action_dense(action)

		x = self.cat([s, a])
		x = self.dense1(x)
		x = self.dense2(x)

		return self.out(x)
