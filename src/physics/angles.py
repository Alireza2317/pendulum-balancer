import numpy as np


def wrap_angle(angle: float) -> float:
	"""Wraps any angle into the [-pi, pi] range."""
	return (angle + np.pi) % (2 * np.pi) - np.pi


def normalize_angle(angle: float) -> float:
	"""
	Takes in a pybullet angle, and normalizes it somehow to make the upright
	position 0, and hanging down 180/-180 degress.
	This is necessary since in pybullet, upright is 180 degrees.
	"""
	return wrap_angle(np.pi - angle)


def normalize_angular_velocities(
	ang_vel1: float, ang_vel2: float
) -> tuple[float, float]:
	"""
	Takes in two angular velocities, and transforms them to match the normalized angles.
	Returns the normalized velocities in order as a tuple.
	"""

	return -ang_vel1, -(ang_vel1 + ang_vel2)


def denormalize_angular_velocities(
	ang_vel_normalized1: float, ang_vel_normalized2: float
) -> tuple[float, float]:
	"""
	Takes in two normalized angular velocities, and transforms them to match the
	original angles.
	Returns the denormalized velocities in order as a tuple.
	"""

	return -ang_vel_normalized1, ang_vel_normalized1 - ang_vel_normalized2


def denormalize_angle(normalized_angle: float) -> float:
	"""
	Takes in a normalized angle, and denormalizes it somehow to make the upright
	position 180 degrees, just the way pybullet expects it to be. Normalized angle
	is somehow that upright is 0 and hanging down is 180/-180 degrees.
	"""

	return wrap_angle(np.pi - normalized_angle)


def rel2abs(angle1: float, angle2_rel: float) -> float:
	"""
	Takes in two relative angles (in radians),
	converts the second angle to the absolute form.
	"""

	return wrap_angle(angle1 + angle2_rel)


def abs2rel(angle1: float, angle2_abs: float) -> float:
	"""
	Takes in two absolute angles (in radians),
	converts the second angle to the relative form.
	"""

	return wrap_angle(angle2_abs - angle1)
