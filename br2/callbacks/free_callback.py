import numpy as np
import matplotlib.pyplot as plt

from elastica import CallBackBaseClass
from elastica.typing import RodType


class FreeCallback(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict, time_interval=None, actuation_ref=None, **kwargs):
        super().__init__(**kwargs)
        self.every = step_skip
        self.time_interval = time_interval
        self.callback_params = callback_params
        self.actuation_ref = actuation_ref

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every != 0:
            return
        if self.time_interval is not None and (
            time < self.time_interval[0] or time > self.time_interval[1]
        ):
            return
        self.callback_params["time"].append(time)
        self.callback_params["step"].append(current_step)
        self.callback_params["position"].append(system.position_collection.copy())
        self.callback_params["velocity"].append(system.velocity_collection.copy())
        self.callback_params["acceleration"].append(
            system.acceleration_collection.copy()
        )
        self.callback_params["omega"].append(system.omega_collection.copy())
        self.callback_params["alpha"].append(system.alpha_collection.copy())
        self.callback_params["director"].append(system.director_collection.copy())
        self.callback_params["external_forces"].append(system.external_forces.copy())
        self.callback_params["external_torques"].append(system.external_torques.copy())
        self.callback_params["internal_forces"].append(system.internal_forces.copy())
        self.callback_params["internal_torques"].append(system.internal_torques.copy())
        self.callback_params["kappa"].append(system.kappa.copy())
        self.callback_params["sigma"].append(system.sigma.copy())
        self.callback_params["lengths"].append(system.lengths.copy())
        self.callback_params["dilatation"].append(system.dilatation.copy())
        self.callback_params["radius"].append(system.radius.copy())
        self.callback_params["com"].append(
            system.compute_position_center_of_mass().copy()
        )
        # self.callback_params["vcom"].append(system.compute_velocity_center_of_mass())

        self.callback_params["actuation"].append(self.actuation_ref())

        self.callback_params["volume"].append(system.volume.copy())
        self.callback_params["alpha_angle"].append(system.alpha_angle.copy())
        self.callback_params["beta_angle"].append(system.beta_angle.copy())
        self.callback_params["delta_turn"].append(system.delta_turn.copy())


class OnlinePlottingRodStatus(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict, time_interval=None):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.time_interval = time_interval
        self.callback_params = callback_params

        self.fig, self.axes = plt.subplots(4, 6, figsize=(12, 8), sharex=True)
        self.plot_position = [
            self.axes[0, 0].plot([], [])[0],
            self.axes[1, 0].plot([], [])[0],
            self.axes[2, 0].plot([], [])[0],
        ]
        self.plot_director = [
            [self.axes[0, 1 + i].plot([], [])[0] for i in range(3)],
            [self.axes[1, 1 + i].plot([], [])[0] for i in range(3)],
            [self.axes[2, 1 + i].plot([], [])[0] for i in range(3)],
        ]
        self.plot_kappa = [
            self.axes[0, 4].plot([], [])[0],
            self.axes[1, 4].plot([], [])[0],
            self.axes[2, 4].plot([], [])[0],
        ]
        self.plot_sigma = [
            self.axes[0, 5].plot([], [])[0],
            self.axes[1, 5].plot([], [])[0],
            self.axes[2, 5].plot([], [])[0],
        ]
        self.plot_characteristics = [
            self.axes[3, 0].plot([], [])[0],
            self.axes[3, 1].plot([], [])[0],
            self.axes[3, 2].plot([], [])[0],
            self.axes[3, 3].plot([], [])[0],
            self.axes[3, 4].plot([], [])[0],
            self.axes[3, 5].plot([], [])[0],
        ]
        self.set_texts()

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every != 0:
            return
        if self.time_interval is not None and (
            time < self.time_interval[0] or time > self.time_interval[1]
        ):
            return
        fig, axes = self.fig, self.axes

        fig.suptitle(f"Time: {time:.2f}, Step: {current_step}")

        self.plot_vector(system.position_collection, self.plot_position)
        self.plot_SO3(system.director_collection, self.plot_director)
        self.plot_vector(system.kappa, self.plot_kappa)
        self.plot_vector(system.sigma, self.plot_sigma)
        self.plot_vector_jagged(
            [
                system.lengths,
                system.dilatation,
                system.radius,
                system.alpha_angle,
                system.beta_angle,
                system.delta_turn,
            ],
            self.plot_characteristics,
        )

        self.release()

    def release(self):
        axes = self.axes
        for ax in axes.ravel():
            ax.relim()
            ax.autoscale_view()
        plt.show(block=False)
        plt.pause(0.01)

    def set_texts(self):
        axes = self.axes

        # Remove x-ticks from all but the bottom row
        for ax in axes[:-1, :].ravel():
            ax.set_xticks([])

        # Set title
        axes[0, 0].set_title("x")
        axes[1, 0].set_title("y")
        axes[2, 0].set_title("z")
        axes[0, 1].set_title("d11")
        axes[1, 1].set_title("d21")
        axes[2, 1].set_title("d31")
        axes[0, 2].set_title("d12")
        axes[1, 2].set_title("d22")
        axes[2, 2].set_title("d32")
        axes[0, 3].set_title("d13")
        axes[1, 3].set_title("d23")
        axes[2, 3].set_title("d33")
        axes[0, 4].set_title("kappa_x")
        axes[1, 4].set_title("kappa_y")
        axes[2, 4].set_title("kappa_z")
        axes[0, 5].set_title("sigma_x")
        axes[1, 5].set_title("sigma_y")
        axes[2, 5].set_title("sigma_z")
        axes[3, 0].set_title("lengths")
        axes[3, 1].set_title("dilatation")
        axes[3, 2].set_title("radius")
        axes[3, 3].set_title("alpha_angle")
        axes[3, 4].set_title("beta_angle")
        axes[3, 5].set_title("delta_turn")

    def plot_vector(self, data, plots):  # plot vector data (3, N)
        N = data.shape[-1]
        indices = np.linspace(0, 1, N)
        for i in range(data.shape[0]):
            pl = plots[i]
            pl.set_xdata(indices)
            pl.set_ydata(data[i])

    def plot_vector_jagged(self, data, plots):  # plot vector data
        for i in range(len(data)):
            N = data[i].shape[-1]
            indices = np.linspace(0, 1, N)
            pl = plots[i]
            pl.set_xdata(indices)
            pl.set_ydata(data[i])

    def plot_SO3(self, data, plots):  # plot so3 data (3, 3, N)
        N = data.shape[-1]
        indices = np.linspace(0, 1, N)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                pl = plots[i][j]
                pl.set_xdata(indices)
                pl.set_ydata(data[i, j])
