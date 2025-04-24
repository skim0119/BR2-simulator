__all__ = ["BlenderRodCallback"]

import bpy
import numpy as np
from elastica import CallBackBaseClass
from elastica.typing import RodType

from bsr import frame_manager
from bsr import BezierSplinePipe

from bsr import RodWithCylinder as Rod

# from bsr import RodWithBox as Rod


class BlenderRodCallback(CallBackBaseClass):
    """
    PyElastica callback to save rod state to Blender.
    """

    def __init__(
        self,
        step_skip: int,
        time_interval: int,
        callback_params=None,
        scale: float = 100.0,
        visualize_alpha_beta=True,
        **kwargs
    ) -> None:
        CallBackBaseClass.__init__(self, **kwargs)
        self.every = step_skip
        self.time_interval = time_interval
        self.key_frame = 0
        self.scale = scale
        self.stop = False

        self.bsr_rod: Rod
        self.bsr_splines_alpha: list[BezierSplinePipe]
        self.bsr_splines_beta: list[BezierSplinePipe]
        self.num_splines = 1

        self.visualize_alpha_beta = visualize_alpha_beta

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
        if np.isnan(system.position_collection).any() or np.isnan(system.radius).any():
            self.stop = True
            return
        if np.isnan(system.alpha_angle).any() or np.isnan(system.beta_angle).any():
            self.stop = True
            return

        # Update rod
        if current_step == 0:
            self.initialize(system)
        else:
            self.update_states(system)
        self.update_keyframes()

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
        self.bsr_rod = Rod(
            system.position_collection * self.scale,
            system.radius * self.scale,
            # system.director_collection,
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
        self.bsr_rod.update_states(
            positions=system.position_collection * self.scale,
            radii=system.radius * self.scale,
            # directors=system.director_collection,
        )

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
            self.bsr_splines_alpha[i].update_states(
                positions=positions * self.scale, radii=radii * self.scale
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
