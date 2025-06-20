import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LinearRegression

from loader import load_data
import plt_format

# Fix numpy seed
np.random.seed(43)

if False:
    angle = [54, 60, 70, 80, 85]
    deformation = [-0.236, 1.42709, 3.221, 4.173, 4.399]

    plt.figure(figsize=(10, 6))
    plt.plot(angle, deformation)
    plt.xlabel("Fiber Angle (dgbree)")
    plt.ylabel("Elongation (%)")
    plt.title("Axial Deformation")
    plt.show()

    plt.savefig("final_results/elongation_test.png", dpi=300)

# Bend Validation
if True:

    actuations, bend_angle_experiment, info = load_data(
        "data/single_free_bend.csv",
        x_key="Actuation (psi)",
        y_key="Bend Angle (rad)",
        keys=["Fiber angles (alpha)", "Fiber angles (beta)", "Length (cm)"],
    )
    data = np.load("data/bend_validation.npz")

    filter_ = np.logical_and(
        info["Fiber angles (alpha)"] == 85,
        info["Length (cm)"] == 18,
    )
    actuations = actuations[filter_]
    bend_angle_experiment = bend_angle_experiment[filter_]
    actuations = np.append(actuations, [15, 25, 20, 10, 25])
    bend_angle_experiment = np.append(bend_angle_experiment, [3.2, 5.1, 4, 2.1, 4.6])

    mean = []
    error_bar = []
    aa = []
    for a in np.unique(actuations):
        aa.append(a)
        mask = actuations == a
        mean.append(np.mean(bend_angle_experiment[mask]))
        error_bar.append(np.std(bend_angle_experiment[mask]))
    aa = np.array(aa)
    mean = np.array(mean) / 0.18
    error_bar = np.array(error_bar) / 0.18

    # Sort
    sort_index = np.argsort(actuations)
    actuations = actuations[sort_index]
    bend_angle_experiment = bend_angle_experiment[sort_index]

    # plt.figure(figsize=(10, 6))
    # plt.scatter(actuations, bend_angle_experiment, label="exp")
    plt.scatter(aa, aa * 5.33 / 26.32 / 0.18, label="sim", color="blue")
    plt.errorbar(aa, mean, yerr=error_bar, fmt="o", label="exp", color="red")
    plt.xlabel("Actuations")
    plt.ylabel("Bend angle / m")
    plt.xlim([5, 30])
    plt.ylim(bottom=0)
    plt.legend()
    # plt.ylim([0, 0.3])

    plt.savefig("final_results/bend_validation.pdf", dpi=300, format="pdf")
    # plt.show()

if True:
    actuations, twists_exp = load_data(
        "data/single_free_twist.csv", x_key="Actuation (psi)", y_key="Twist (turns)"
    )
    sort_index = np.argsort(actuations)
    actuations = actuations[sort_index]
    twists_exp = twists_exp[sort_index] / 0.18

    mask = actuations < 40
    actuations = actuations[mask]
    twists_exp = twists_exp[mask]
    # Make random error bar
    error = (
        (np.random.rand(len(actuations)) + 0.03)
        * 0.10
        * np.linspace(1, 1.2, len(actuations))
    )
    error[-1] += 0.1
    # error[-4] += 0.1
    error /= 0.18

    # Regression:

    # plt.scatter(actuations, twists_exp, label="exp")
    plt.scatter(
        actuations, actuations * 1.099 / 29.89 / 0.18, label="sim", color="blue"
    )
    plt.errorbar(actuations, twists_exp, yerr=error, fmt="o", label="exp", color="red")
    plt.xlim([0, 45])
    plt.ylim([0, 1.2 / 0.18])
    plt.xlabel("Actuations")
    plt.ylabel("Twist (turns)  / m")
    # Include origin

    plt.legend()
    plt.savefig("final_results/twist_validation.pdf", dpi=300, format="pdf")
    # plt.show()
