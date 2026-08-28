import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.physics.env import EnvState


@dataclass(frozen=True)
class Transition:
	state: EnvState
	action: float
	reward: float
	next_state: EnvState
	done: bool


@dataclass(frozen=True)
class Batch:
	states: npt.NDArray[np.float32]
	actions: npt.NDArray[np.float32]
	rewards: npt.NDArray[np.float32]
	next_states: npt.NDArray[np.float32]
	dones: npt.NDArray[np.float32]


class IBuffer(ABC):
	@abstractmethod
	def add(self, transition: Transition) -> None:
		"""Stores a transition in the buffer."""
		pass

	@abstractmethod
	def sample(self, batch_size: int) -> Batch:
		"""Samples a batch and returns arrays formatted for training."""
		pass

	@abstractmethod
	def __len__(self) -> int:
		"""Returns the number of transitions currently stored."""
		pass

	@abstractmethod
	def clear(self) -> None:
		"""Clears all the transitions currently stored in the buffer."""
		pass


class UniformReplayBuffer(IBuffer):
	def __init__(self, capacity: int) -> None:
		self._buffer: deque[Transition] = deque(maxlen=capacity)

	def add(self, transition: Transition) -> None:
		self._buffer.append(transition)

	def sample(self, batch_size: int) -> Batch:
		batch: list[Transition] = random.sample(self._buffer, k=batch_size)

		return Batch(
			states=np.array([t.state.nparray() for t in batch], dtype=np.float32),
			actions=np.array([t.action for t in batch], dtype=np.float32),
			rewards=np.array([t.reward for t in batch], dtype=np.float32),
			next_states=np.array(
				[t.next_state.nparray() for t in batch], dtype=np.float32
			),
			dones=np.array([t.done for t in batch], dtype=np.float32),
		)

	def __len__(self) -> int:
		return len(self._buffer)

	def clear(self) -> None:
		self._buffer.clear()
