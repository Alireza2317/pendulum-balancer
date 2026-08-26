from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class EnvState:
	cart_x: float
	cart_x_velocity: float
	pole1_angle: float
	pole1_angular_velocity: float
	pole2_angle: float
	pole2_angular_velocity: float

	def nparray(self) -> npt.NDArray[np.float32]:
		return np.array(
			[
				self.cart_x,
				self.cart_x_velocity,
				self.pole1_angle,
				self.pole1_angular_velocity,
				self.pole2_angle,
				self.pole2_angular_velocity,
			],
			dtype=np.float32,
		)
