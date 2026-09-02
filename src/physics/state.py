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
		# Angles are fed as (sin, cos) rather than raw radians.
		# Raw angle has a discontinuity at +-pi (physically the same)
		# But represented as two far-apart numbers
		return np.array(
			[
				self.cart_x,
				self.cart_x_velocity,
				np.sin(self.pole1_angle),
				np.cos(self.pole1_angle),
				self.pole1_angular_velocity,
				np.sin(self.pole2_angle),
				np.cos(self.pole2_angle),
				self.pole2_angular_velocity,
			],
			dtype=np.float32,
		)

	def __repr__(self) -> str:
		rp: str = "EnvState(\n"
		rp += f"\tcart_x={self.cart_x:+5.3f}\n"
		rp += f"\tcart_velocity={self.cart_x_velocity:+5.3f}\n"
		rp += f"\tpole1_angle(deg)={np.rad2deg(self.pole1_angle):+5.3f}\n"
		rp += f"\tpole2_angle(deg)={np.rad2deg(self.pole2_angle):+5.3f}\n"
		rp += f"\tpole1_velocity={self.pole1_angular_velocity:+5.3f}\n"
		rp += f"\tpole2_velocity={self.pole2_angular_velocity:+5.3f}\n"
		rp += ")"

		return rp
