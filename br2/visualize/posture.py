import numpy as np

import matplotlib.pyplot as plt
from abc import abstractmethod, ABC

from .strain import PlotterContext


class PosturePlot(PlotterContext):
    def plot(self, time, position, directors, label=""):
        fig, axs = self.fig, self.axes

        T = len(time)
        N = position.shape[-1] - 1

        axs[2, 0].set_xlabel("Time")
        axs[2, 1].set_xlabel("Time")
        axs[2, 2].set_xlabel("Time")
        axs[2, 3].set_xlabel("Time")

        # Plot position
        axs[0, 0].set_title("Position")
        axs[0, 0].set_ylabel("x")
        axs[1, 0].set_ylabel("y")
        axs[2, 0].set_ylabel("z")
        for i in range(3):
            # axs[i, 0].plot(time, position[:, i, -1], label=label)
            axs[i, 0].plot(time, position[:, i], label=label)

        # Plot directors
        axs[0, 1].set_title("Director (normal)")
        axs[0, 2].set_title("Director (binormal)")
        axs[0, 3].set_title("Director (tangent)")
        for i in range(3):
            for j in range(3):
                ax = axs[i, j + 1]
                # ax.plot(time, directors[:, i, j, -1], label=label)
                ax.plot(time, directors[:, j, i], label=label)


class FixPosturePlot(PlotterContext):
    def plot(self, position, directors, time_index, label=""):
        fig, axs = self.fig, self.axes

        axs[2, 0].set_xlabel("Nodes")
        axs[2, 1].set_xlabel("Elements")
        axs[2, 2].set_xlabel("Elements")
        axs[2, 3].set_xlabel("Elements")

        # Plot position
        axs[0, 0].set_title("Position")
        axs[0, 0].set_ylabel("x")
        axs[1, 0].set_ylabel("y")
        axs[2, 0].set_ylabel("z")
        for i in range(3):
            axs[i, 0].plot(position[time_index, i], label=label)

        # Plot directors
        axs[0, 1].set_title("Director (normal)")
        axs[0, 2].set_title("Director (binormal)")
        axs[0, 3].set_title("Director (tangent)")
        for i in range(3):
            for j in range(3):
                ax = axs[i, j + 1]
                ax.plot(directors[time_index, j, i], label=label)
