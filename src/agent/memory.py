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


class ReplayBuffer:
	def __init__(self, capacity: int) -> None:
		self._buffer: deque[Transition] = deque(maxlen=capacity)

	def add(self, transition: Transition) -> None:
		self._buffer.append(transition)

	def sample(self, batch_size: int) -> npt.NDArray:
		return np.random.choice(
			np.asarray(self._buffer), size=batch_size, replace=False
		)

	def __len__(self) -> int:
		return len(self._buffer)
