"""Headless regression tests for episode bookkeeping; no network construction."""

import unittest
from unittest.mock import Mock

from src.agent.agent import TD3Agent
from src.agent.memory import UniformReplayBuffer
from src.config import Config
from src.physics.env import DoublePendulumEnv
from src.physics.state import EnvState
from src.trainer.trainer import TD3Trainer


class FakeEnvironment:
	"""Produce predictable transitions and fail on a chosen step."""

	def __init__(self, fail_at: int | None):
		self.fail_at = fail_at
		self.calls = 0

	def reset(self) -> EnvState:
		self.calls = 0
		return EnvState(0, 0, 0, 0, 0, 0)

	def step(self, action: float):
		self.calls += 1
		if self.fail_at is not None and self.calls > self.fail_at:
			raise AssertionError("Trainer stepped after termination")
		state = EnvState(self.calls / 100, 0, 0, 0, 0, 0)
		done = self.calls == self.fail_at
		return state, 1.0, done, {}


class EpisodeBookkeepingTests(unittest.TestCase):
	def check_episode(self, fail_at: int | None, expected_steps: int):
		cfg = Config(
			batch_size=2,
			buffer_warmup_size=20,
			buffer_maxsize=20,
			max_episode_steps=10,
		)
		fake = FakeEnvironment(fail_at)
		# Spec mocks expose the production interfaces without constructing
		# a PyBullet connection or any TensorFlow networks.
		env = Mock(spec=DoublePendulumEnv)
		env.reset.side_effect = fake.reset
		env.step.side_effect = fake.step
		agent = Mock(spec=TD3Agent)
		agent.get_action.return_value = 0.0
		replay = UniformReplayBuffer(cfg.buffer_maxsize)
		trainer = TD3Trainer(cfg, env, agent, replay)
		trainer.noise = Mock()
		trainer.noise.sample.return_value = 0.0

		episode_result, agent_metric, _ = trainer.run_episode()

		self.assertGreaterEqual(episode_result.balance_fraction, 0)
		self.assertLessEqual(episode_result.balance_fraction, 1)
		self.assertEqual(episode_result.steps, expected_steps)
		self.assertEqual(fake.calls, expected_steps)
		self.assertEqual(env.step.call_count, expected_steps)
		self.assertEqual(agent.get_action.call_count, expected_steps)
		self.assertEqual(len(replay), expected_steps)
		self.assertEqual(episode_result.total_reward, expected_steps)

		expected_success_ratio = 0.0 if fail_at is not None else 1.0
		self.assertEqual(
			trainer.curriculum.success_ratio,
			expected_success_ratio,
		)
		self.assertEqual(
			(
				agent_metric.actor_loss,
				agent_metric.critic_loss,
				agent_metric.q_vals_avg,
			),
			(0.0, 0.0, 0.0),
		)
		agent.train_step.assert_not_called()
		env.reset.assert_called_once()

		# Sample every transition; ordering does not matter. The terminal
		# transition must be retained, and a time cap must not invent one.
		batch = replay.sample(expected_steps)
		self.assertEqual(int(batch.dones.sum()), int(fail_at is not None))
		self.assertAlmostEqual(
			float(batch.next_states[:, 0].max()), expected_steps / 100
		)

	def test_failure_on_first_step(self):
		self.check_episode(fail_at=1, expected_steps=1)

	def test_failure_on_third_step(self):
		self.check_episode(fail_at=3, expected_steps=3)

	def test_failure_on_last_allowed_step(self):
		self.check_episode(fail_at=10, expected_steps=10)

	def test_time_limit_without_failure(self):
		self.check_episode(fail_at=None, expected_steps=10)


if __name__ == "__main__":
	unittest.main()
