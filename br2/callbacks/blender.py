__all__ = ["BlenderRodCallback"]

import bpy
import numpy as np
from matplotlib import cm
from matplotlib import colors
from elastica import CallBackBaseClass
from elastica.typing import RodType

from bsr import frame_manager
from bsr import BezierSplinePipe

# from bsr import RodWithCylinder as Rod
# from bsr import RodWithBox as Rod  # Need to pass director
from bsr import BezierSplinePipe as Rod


ZOFFSET = 0.35


class BlenderRodCallback(CallBackBaseClass):
    """
    PyElastica callback to save rod state to Blender.
    """

    def __init__(
        self,
        step_skip: int,
        time_interval: int,
        callback_params=None,
        scale: float = 10.0,
        visualize_alpha_beta=True,
        is_ring=False,
        actuation_ref=None,
        actuation_max=100,
        cmap="Oranges",
        **kwargs,
    ) -> None:
        # CallBackBaseClass.__init__(self, **kwargs)
        self.every = step_skip
        self.time_interval = time_interval
        self.key_frame = 0
        self.scale = scale
        self.stop = False

        self.initialized = False
        self.bsr_rod: Rod
        self.bsr_splines_alpha: list[BezierSplinePipe]
        self.bsr_splines_beta: list[BezierSplinePipe]
        self.num_splines = 1

        self.visualize_alpha_beta = visualize_alpha_beta
        self.is_ring = is_ring

        self.actuation_ref = actuation_ref  # if none, no action input.
        self.actuation_max = actuation_max
        self.cmap = cmap

    def make_callback(
        self, system: RodType, time: np.floating, current_step: int
    ) -> None:
        # Stopping criteria
        if self.stop or current_step % self.every != 0:
            return
        if self.time_interval is not None and (
            time < self.time_interval[0] or time > self.time_interval[1]
        ):
            return

        # Halting if there are NaNs in the system
        if (
            np.isnan(system.position_collection).any()
            or np.isnan(system.radius).any()
            or np.isnan(system.director_collection).any()
        ):
            self.stop = True
            return

        if hasattr(system, "alpha_angle") and (
            np.isnan(system.alpha_angle).any() or np.isnan(system.beta_angle).any()
        ):
            self.stop = True
            return

        # Update rod
        if not self.initialized:
            self.initialize(system)
            self.initialized = True
        else:
            self.update_states(system)
        # Update action
        if self.actuation_ref is None:
            pressure = 0.0
        else:
            pressure = self.actuation_ref()
        rgba = self.map_pressure_to_rgba(
            pressure, x_max=self.actuation_max, cmap=self.cmap
        )
        self.bsr_rod.update_material(color=np.array(rgba))

        self.update_keyframes()

    def map_pressure_to_rgba(
        self, x: float, x_max: float, cmap: str
    ) -> tuple[float, float, float, float]:
        """
        Map a value x ∈ [0, x_max] to an RGBA color in the 'Oranges' colormap.

        Parameters
        ----------
        x : float
            Input value, should satisfy 0 <= x <= x_max.
        x_max : float
            Maximum value of the range. Must be > 0.

        Returns
        -------
        rgba : tuple of four floats
            Corresponding (r, g, b, a) color, each in [0, 1].
        """
        if x_max <= 0:
            raise ValueError("x_max must be positive")
        x_clipped = max(0.0, min(x, x_max))
        norm = colors.Normalize(vmin=0.0, vmax=x_max)
        cmap = cm.get_cmap(cmap)
        rgba = cmap(norm(x_clipped))
        return rgba  # rgba is (r, g, b, a)

    @staticmethod
    def find_helix(
        position,
        lengths,
        radius,
        initial_radius,
        director_normal,
        tangent,
        fiber_angle,
        helix_radius_ratio=1.0 / 20,
        num_spline_resolution=32,
    ):
        radii = np.repeat(initial_radius * helix_radius_ratio, num_spline_resolution)

        s = position[:, 1:] - position[:, :-1]
        # tangent = s / np.linalg.norm(s, axis=0, keepdims=True)
        normal = director_normal - (
            (tangent * director_normal).sum(axis=0, keepdims=True) * tangent
        )
        normal /= np.linalg.norm(normal, axis=0, keepdims=True)
        binormal = np.cross(tangent, normal, axis=0)

        l = np.linspace(0, lengths, num_spline_resolution, endpoint=False)
        dphi = np.cumsum(lengths * np.tan(fiber_angle) / radius)[None, :-1]
        phi = l * np.tan(fiber_angle) / radius
        phi[:, 1:] += dphi  # Offset
        points = (
            position[:, :-1, None]
            + radius[None, :, None]
            * (normal[..., None] * np.cos(phi).T + binormal[..., None] * np.sin(phi).T)
            + tangent[..., None] * l.T
        )
        points = points.reshape(3, -1)

        # import matplotlib.pyplot as plt
        # plt.plot(points[0], label='x')
        ##plt.plot(points[1], label='y')
        # plt.plot(points[2], label='z')
        # plt.legend()
        # plt.show()

        # breakpoint()

        return points, radii

    def initialize(self, system) -> None:
        positions = system.position_collection + np.array([0.0, 0.0, ZOFFSET])[:, None]
        if self.is_ring:
            positions = np.append(positions, positions[..., :1], axis=1)
        self.bsr_rod = Rod(
            positions=positions * self.scale,
            radii=system.radius * self.scale,
            # directors=system.director_collection,
            downsample_num_element=10,
        )

        self.bsr_splines_alpha = []
        self.bsr_splines_beta = []

        if not self.visualize_alpha_beta:
            return

        # Add alpha angle
        for i in range(self.num_splines):
            positions, radii = self.find_helix(
                system.position_collection,
                system.lengths,
                system.radius,
                system.initial_radius,
                system.director_collection[0],
                system.tangents,
                system.alpha_angle,
            )
            self.bsr_splines_alpha.append(
                BezierSplinePipe(
                    positions=positions * self.scale, radii=radii * self.scale
                )
            )

        # Add beta angle
        for i in range(self.num_splines):
            positions, radii = self.find_helix(
                system.position_collection,
                system.lengths,
                system.radius,
                system.initial_radius,
                system.director_collection[0],
                system.tangents,
                system.beta_angle,
            )
            self.bsr_splines_beta.append(
                BezierSplinePipe(
                    positions=positions * self.scale, radii=radii * self.scale
                )
            )

    def update_states(self, system) -> None:
        positions = system.position_collection + np.array([0.0, 0.0, ZOFFSET])[:, None]
        if self.is_ring:
            positions = np.append(positions, positions[..., :1], axis=1)
        self.bsr_rod.update_states(
            positions=positions * self.scale,
            radii=system.radius * self.scale,
            # directors=system.director_collection,
        )

        if not self.visualize_alpha_beta:
            return

        # Add alpha angle
        for i in range(self.num_splines):
            positions, radii = self.find_helix(
                system.position_collection + np.array([0.0, 0.0, ZOFFSET])[:, None],
                system.lengths,
                system.radius,
                system.initial_radius,
                system.director_collection[0],
                system.tangents,
                system.alpha_angle,
            )
            self.bsr_splines_alpha[i].update_states(
                positions=positions * self.scale, radii=radii * self.scale
            )

        # Add beta angle
        for i in range(self.num_splines):
            positions, radii = self.find_helix(
                system.position_collection + np.array([0.0, 0.0, ZOFFSET])[:, None],
                system.lengths,
                system.radius,
                system.initial_radius,
                system.director_collection[0],
                system.tangents,
                system.beta_angle,
            )
            self.bsr_splines_beta[i].update_states(
                positions=positions * self.scale, radii=radii * self.scale
            )

    def update_keyframes(self) -> None:
        self.bsr_rod.update_keyframe(self.key_frame)
        if self.visualize_alpha_beta:
            for spline in self.bsr_splines_alpha:
                spline.update_keyframe(self.key_frame)
            for spline in self.bsr_splines_beta:
                spline.update_keyframe(self.key_frame)
        self.key_frame += 1
        frame_manager.current_frame = self.key_frame
