import numpy as np
from numba import njit
from numpy import sqrt
import functools
from itertools import permutations
from elastica.utils import perm_parity
from elastica._linalg import _batch_matvec, _batch_norm
from surface_connection_parallel_rod_numba import (
    SurfaceJointSideBySide,
    TipToTipStraightJoint
)

def glue_rods_surface_connection(sim, rod1, rod2, k, nu, kt):
        rod1_pos = 0.5 * (
                rod1.position_collection[..., 1:]
                + rod1.position_collection[..., :-1]
        )#找出位置中间值
        rod2_pos = 0.5 * (
                rod2.position_collection[..., 1:]
                + rod2.position_collection[..., :-1]
        )
        rod1_Q = rod1.director_collection
        rod2_Q = rod2.director_collection
        distance = _batch_norm(rod2_pos - rod1_pos)
        assert np.allclose(
            distance, rod1.radius + rod2.radius
        ), "Not all elements are touching eachother"
        connection_lab = (rod2_pos - rod1_pos) / distance
        rod1_rd2_local = _batch_matvec(rod1_Q, connection_lab)  # local frame
        rod2_rd2_local = _batch_matvec(rod2_Q, -connection_lab)  # local frame

        sim.connect(
            first_rod=rod1, second_rod=rod2
        ).using(
            SurfaceJointSideBySide,
            k=k,
            nu=nu,
            kt=kt,
            rd1_local=rod1_rd2_local,
            rd2_local=rod2_rd2_local,
            stability_check=False
        ) 