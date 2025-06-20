import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from abc import abstractmethod, ABC


class PlotterContext(ABC):
    def __init__(self, fig, axes):
        self.fig = fig
        self.axes = axes

    def finalize_plot(self):
        for ax in self.axes.ravel():
            ax.legend(loc="best")
            ax.grid()

    def savefig(self, path):
        self.fig.savefig(path, dpi=300)

    def show(self):
        plt.show()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.close(self.fig)

    @abstractmethod
    def plot(self, *data, label=None):
        """
        Plot the data on the axes
        """
        pass


class StrainPosturePlot(PlotterContext):
    def plot(self, time, position, directors, shear, kappa, label=""):
        fig, axs = self.fig, self.axes

        T = len(time)
        N = position.shape[-1] - 1

        axs[2, 0].set_xlabel("Element")
        axs[2, 1].set_xlabel("Element")
        axs[2, 2].set_xlabel("Element")
        axs[2, 3].set_xlabel("Element")
        axs[2, 4].set_xlabel("Element")
        axs[2, 5].set_xlabel("Element")

        # Plot position
        axs[0, 0].set_title("Position")
        axs[0, 0].set_ylabel("x")
        axs[1, 0].set_ylabel("y")
        axs[2, 0].set_ylabel("z")
        for i in range(3):
            axs[i, 0].plot(position[-1, i, :], label=label)

        # Plot shear
        axs[0, 1].set_title("Shear")
        for i in range(3):
            axs[i, 1].plot(shear[-1, i, :], label=label)

        # Plot kappa
        axs[0, 2].set_title("Kappa")
        for i in range(3):
            axs[i, 2].plot(kappa[-1, i, :], label=label)

        # Plot directors
        axs[0, 3].set_title("Director (normal)")
        axs[0, 4].set_title("Director (binormal)")
        axs[0, 5].set_title("Director (tangent)")
        for i in range(3):
            for j in range(3):
                ax = axs[i, j + 3]
                ax.plot(directors[-1, j, i, :], label=label)


class StrainTimePlot(PlotterContext):
    def plot(self, time, position, directors, shear, kappa, label=""):
        fig, axs = self.fig, self.axes

        T = len(time)
        N = position.shape[-1] - 1

        axs[2, 0].set_xlabel("Time")
        axs[2, 1].set_xlabel("Time")
        axs[2, 2].set_xlabel("Time")
        axs[2, 3].set_xlabel("Time")
        axs[2, 4].set_xlabel("Time")
        axs[2, 5].set_xlabel("Time")

        # Plot position
        axs[0, 0].set_title("Position")
        axs[0, 0].set_ylabel("x")
        axs[1, 0].set_ylabel("y")
        axs[2, 0].set_ylabel("z")
        for i in range(3):
            axs[i, 0].plot(time, position[:, i, -1], label=label)

        # Plot shear
        axs[0, 1].set_title("Shear")
        for i in range(3):
            axs[i, 1].plot(time, shear[:, i, -1], label=label)

        # Plot kappa
        axs[0, 2].set_title("Kappa")
        for i in range(3):
            axs[i, 2].plot(time, kappa[:, i, -1], label=label)

        # Plot directors
        axs[0, 3].set_title("Director (normal)")
        axs[0, 4].set_title("Director (binormal)")
        axs[0, 5].set_title("Director (tangent)")
        for i in range(3):
            for j in range(3):
                ax = axs[i, j + 3]
                ax.plot(time, directors[:, j, i, -1], label=label)


class KappaPlot(PlotterContext):
    def __init__(self):
        # Create Two 3D plot on left, regular plot on right
        fig = plt.figure(figsize=(12, 6), num="Kappa")
        ax1 = fig.add_subplot(131, projection="3d")
        ax2 = fig.add_subplot(132, projection="3d")
        ax3 = fig.add_subplot(133)
        axes = np.array([[ax1, ax2, ax3]])
        super().__init__(fig, axes)

    def plot(self, time, position, directors, shear, kappa, label=""):
        position = 0.5 * (position[:, :, :-1] + position[:, :, 1:])

        fig, axs = self.fig, self.axes

        T = len(time)
        N = position.shape[-1] - 1

        quiver_length = 0.010

        # 3D plot of position and directors as a quiver
        ax = axs[0, 0]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        scatter = ax.scatter(
            position[-1, 0, :], position[-1, 1, :], position[-1, 2, :], label=label
        )
        color = scatter.get_facecolor()[0].tolist()
        ax.quiver(
            position[-1, 0, :],
            position[-1, 1, :],
            position[-1, 2, :],
            directors[-1, 0, 0, :],
            directors[-1, 0, 1, :],
            directors[-1, 0, 2, :],
            length=quiver_length,
            color=color,
            alpha=0.8,
        )
        ax.quiver(
            position[-1, 0, :],
            position[-1, 1, :],
            position[-1, 2, :],
            directors[-1, 1, 0, :],
            directors[-1, 1, 1, :],
            directors[-1, 1, 2, :],
            length=quiver_length,
            color=color,
            alpha=0.1,
        )
        ax.set_aspect("equal")

        # 3D plot of position and directors as a quiver
        ax = axs[0, 1]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        scatter = ax.scatter(
            position[-1, 0, :], position[-1, 1, :], position[-1, 2, :], label=label
        )
        color = scatter.get_facecolor()[0].tolist()
        ax.quiver(
            position[-1, 0, :],
            position[-1, 1, :],
            position[-1, 2, :],
            directors[-1, 2, 0, :],
            directors[-1, 2, 1, :],
            directors[-1, 2, 2, :],
            length=quiver_length,
            color=color,
            alpha=1.0,
        )
        ax.set_aspect("equal")
