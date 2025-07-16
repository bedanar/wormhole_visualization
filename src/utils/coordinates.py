"""A module to compute z-coordinates of a wormhole surface based on the Morris-Thorne model."""

import numpy as np
from typing import Union


def z_value(r: Union[float, np.ndarray], b: float) -> np.ndarray:
    """
    Computes the z-coordinate of the wormhole surface for a regular trajectory.

    The calculation is based on the Morris-Thorne traversable wormhole metric:
        z(r) = b * arccosh(r / b)

    Args:
        r (float or np.ndarray): Radial coordinate(s).
        b (float): Throat radius of the wormhole.

    Returns:
        np.ndarray: Corresponding z-coordinate(s).
    """
    r = np.asarray(r)
    return b * np.arccosh(np.clip(r / b, 1, None))


def z_value_for_singularity(r: Union[float, np.ndarray], b: float) -> np.ndarray:
    """
    Computes the z-coordinate of the wormhole surface for singular trajectories.

    Like `z_value`, but flips the second half of the z-array to simulate a mirror-like
    geometry across the wormhole throat.

    Args:
        r (float or np.ndarray): Radial coordinate(s).
        b (float): Throat radius of the wormhole.

    Returns:
        np.ndarray: Modified z-coordinate(s), symmetric about the throat.
    """
    r = np.asarray(r)
    z = b * np.arccosh(np.clip(r / b, 1, None))

    # Flip the second half to negative z (symmetric reflection)
    midpoint = z.size // 2
    z[midpoint:] *= -1

    return z
