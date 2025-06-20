import os
import sys, logging
import json
import numpy as np
from dmosopt import dmosopt

import tempfile
from pydantic_core import from_json
import matplotlib.pyplot as plt

from br2.configurations import RodLibrary

from loader import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tag = "validation"


def objective_twist(youngs_modulus, density, twist_scale, **kwargs):
    from run_free_twist import single_free_twist

    # Load experimental data
    actuations, twists = load_data(
        "data/single_free_twist.csv", x_key="Actuation (psi)", y_key="Twist (turns)"
    )

    # Setup configuration for simulation
    assembly_config_path = "assembly/free_twist_18cm_60_0.json"

    # Set New Configuration
    original_database_path = "rod_library/standard_18.json"
    with open(original_database_path, "r") as f:
        json_data = f.read()
    rod_library = RodLibrary.model_validate(from_json(json_data))

    rod_library.DefaultParams.youngs_modulus = youngs_modulus
    rod_library.DefaultParams.shear_modulus = youngs_modulus / 3.0
    rod_library.DefaultParams.density = density

    rod_library.Rods["RodLeftTwist"]["twist_scale"] = twist_scale
    rod_library.Rods["RodRightTwist"]["twist_scale"] = twist_scale

    new_json_data = rod_library.model_dump()

    # Run
    loss = 0
    tfile = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")
    rod_database_path = tfile.name
    json.dump(new_json_data, tfile)
    tfile.flush()

    for actuation, twist_experimental in zip(actuations, twists):
        twist_simulated = single_free_twist(
            actuation,
            rod_database_path,
            assembly_config_path,
            tag,
        )
        loss += np.abs(twist_simulated - twist_experimental)

    tfile.close()

    return loss


def objective_bend(youngs_modulus, density, moment_scale, **kwargs):
    from run_free_bend import single_free_bend

    #
    Load experimental data
    actuations, bend_angle, info = load_data(
        "data/single_free_bend.csv",
        x_key="Actuation (psi)",
        y_key="Bend Angle (rad)",
        keys=["Fiber angles (alpha)", "Fiber angles (beta)", "Length (cm)"],
    )

    # Setup configuration for simulation
    assembly_config_path = "assembly/free_bend_18cm_85_85.json"

    # Set New Configuration
    original_database_path = "rod_library/standard_18.json"
    with open(original_database_path, "r") as f:
        json_data = f.read()
    rod_library = RodLibrary.model_validate(from_json(json_data))

    rod_library.DefaultParams.youngs_modulus = youngs_modulus
    rod_library.DefaultParams.shear_modulus = youngs_modulus / 3.0
    rod_library.DefaultParams.density = density

    # Run
    loss = 0
    for actuation, bend_angle_experimental, alpha, beta, length in zip(
        actuations,
        bend_angle,
        info["Fiber angles (alpha)"],
        info["Fiber angles (beta)"],
        info["Length (cm)"],
    ):
        if length != 18:
            continue
        rod_library.Rods["RodBend"]["alpha"] = alpha.item()
        rod_library.Rods["RodBend"]["beta"] = -beta.item()
        rod_library.Rods["RodBend"]["moment_scale"] = moment_scale.item()

        # Create new configuration
        new_json_data = rod_library.model_dump()
        tfile = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")
        rod_database_path = tfile.name
        json.dump(new_json_data, tfile)
        tfile.flush()

        bend_angle_simulated, _ = single_free_bend(
            actuation,
            rod_database_path,
            assembly_config_path,
            tag,
        )
        loss += np.abs(bend_angle_simulated - bend_angle_experimental)

    tfile.close()

    return loss


def obj_fun(pp):
    """Objective function to be minimized."""
    l1 = objective_twist(**pp)
    l2 = objective_bend(**pp)
    logger.info(f"{l1=}, {l2=}")
    res = np.array(
        [
            l1,
            l2,
        ]
    )
    logger.info(f"Iter: \t pp:{pp}, result:{res}")
    return res


if __name__ == "__main__":
    # actuations, twists = load_data("data/single_free_twist.csv")
    # plt.plot(actuations, twists, 'o', label='experimental data')
    # plt.xlabel('Actuation')
    # plt.ylabel('Twist')
    # plt.show()
    # sys.exit()
    h5_path = os.path.join(f"opt_{tag}", "dmosopt_results.h5")
    opt_id = "optimize_free"
    space = {
        "youngs_modulus": [1.0e6, 5.0e7],
        "density": [1100.0, 2300.0],
        "twist_scale": [1e-4, 1e4],
        "moment_scale": [1e-6, 3e-1],
    }
    objective_names = ["loss_twist_match", "loss_bend_match"]

    # Create an optimizer
    dmosopt_params = {
        "opt_id": opt_id,
        "obj_fun_name": "optimize_free.obj_fun",
        "obj_fun_init_args": {},
        "problem_parameters": {},
        "space": space,
        "objective_names": objective_names,
        "population_size": 20,
        "num_generations": 200,
        "initial_maxiter": 10,
        "optimizer_name": "nsga2",
        "termination_conditions": True,
        "n_initial": 10,
        "n_epochs": 15,
        "file_path": h5_path,
        "save": True,
        "save_surrogate_evals": True,
        "save_optimizer_params": True,
    }

    best = dmosopt.run(dmosopt_params, verbose=True)

    from fdmosopt import Dmosopt

    dmosopt = Dmosopt(dmosopt_config=dmosopt_params)
    best = dmosopt.get_best()
    bestx = best["x"]
    besty = best["y"]

    optimizer_data = dmosopt.load_h5_optimizer_data(opt_id=opt_id)
    results = dmosopt.load_h5(opt_id=opt_id)
    x = results["parameters"].values
    y = results["objectives"]["loss"].values

    # plot results
    plt.scatter(x[:, 0], x[:, 1], s=y**2, label="points")
    plt.scatter(
        bestx["youngs_modulus"],
        bestx["density"],
        label="best points",
        marker="x",
        s=y.min() ** 2,
    )
    plt.xlabel("Young's Modulus")
    plt.ylabel("Density")
    plt.legend()
    p
