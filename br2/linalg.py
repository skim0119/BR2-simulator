import warnings
import numpy as np
from numpy import cos, sin, sqrt
import numba
from numba import njit
from elastica.joint import FreeJoint
from elastica.utils import Tolerance

# Join the two rods
from elastica._linalg import (
    _batch_norm,
    _batch_cross,
    _batch_matvec,
    _batch_dot,
    _batch_matmul,
    _batch_matrix_transpose,
)
from elastica.interaction import (
    elements_to_nodes_inplace,
    node_to_element_position,
    node_to_element_velocity,
)

from elastica._rotations import _inv_skew_symmetrize


@njit(cache=True)
def _single_inv_rotate(director):
    vector = np.empty((3))

    vector[0] = director[2, 1] - director[1, 2]
    vector[1] = director[0, 2] - director[2, 0]
    vector[2] = director[1, 0] - director[0, 1]
    trace = director[0, 0] + director[1, 1] + director[2, 2]

    rtol = 1e-5
    atol = 1e-8
    if np.abs(trace - 3) <= (atol + rtol * 3):
        # if np.isclose(trace, 3):
        multiplier = 0.5 - (trace - 3.0) / 12.0
        vector *= multiplier
        # warnings.warn("Misalignment trace close to 3", RuntimeWarning)
    elif np.abs(trace + 1) <= (atol + rtol):
        # elif np.isclose(trace, -1):
        a = np.argmax(np.diag(director))
        b = (a + 1) % 3
        c = (a + 2) % 3
        s = np.sqrt(director[a, a] - director[b, b] - director[c, c] + 1)
        v = np.array(
            [
                s / 2,
                (1 / (2 * s)) * (director[b, a] + director[a, b]),
                (1 / (2 * s)) * (director[c, a] + director[a, c]),
            ]
        )
        norm_v = np.sqrt(np.sum(v * v))
        vector = np.pi * v / norm_v
    else:
        theta = np.arccos(0.5 * trace - 0.5)
        multiplier = -0.5 * theta / np.sin(theta + 1e-14)
        vector *= multiplier

    return vector

