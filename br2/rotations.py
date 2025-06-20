import numpy as np
from numpy import sin
from numpy import arccos

from numba import njit


@njit(cache=True)
def inv_rotate(director_collection):
    """
    Calculated rate of change using Rodrigues' formula

    Parameters
    ----------
    director_collection : The collection of frames/directors at every element,
    numpy.ndarray of shape (dim, dim, n)

    Returns
    -------
    vector_collection : The collection of axes around which the body rotates
    numpy.ndarray of shape (dim, n)

    Note
    ----
    TODO: Benchmark missing

    """
    blocksize = director_collection.shape[2]
    vector_collection = np.empty((3, blocksize))

    for k in range(blocksize):
        vector_collection[0, k] = (
            director_collection[2, 1, k] - director_collection[1, 2, k]
        )
        vector_collection[1, k] = (
            director_collection[0, 2, k] - director_collection[2, 0, k]
        )
        vector_collection[2, k] = (
            director_collection[1, 0, k] - director_collection[0, 1, k]
        )
        trace = (
            director_collection[0, 0, k]
            + director_collection[1, 1, k]
            + director_collection[2, 2, k]
        )

        # TODO HARDCODED bugfix has to be changed. Remove 1e-14 tolerance
        theta = arccos(0.5 * trace - 0.5 - 1e-10)
        scale = 0.5 * theta / sin(theta + 1e-14)

        # epsilon = 1e-10
        # if trace > 3 - epsilon:
        #     scale = 0.5 - (trace - 3.0) / 12.0
        # elif 3 - epsilon > trace > -1 + epsilon:
        #     theta = arccos(0.5 * (trace - 1))
        #     scale = theta / (2 * sin(theta))
        # else:
        #     # TODO: can be better
        #     scale = 0.0
        # TODO: minus sign??
        vector_collection[0, k] *= scale
        vector_collection[1, k] *= scale
        vector_collection[2, k] *= scale

    return vector_collection
