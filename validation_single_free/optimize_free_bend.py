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
from run_free_bend import single_free_bend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tag = "bend_validation"


def objective(moment_scale):

    # Load experimental data
    actuations, bend_angle, info = load_data(
        "data/single_free_bend.csv",
        x_key="Actuation (psi)",
        y_key="Bend Angle (rad)",
        keys=["Fiber angles (alpha)", "Fiber angles (beta)", "Length (cm)"],
    )

    # Setup configuration for simulation
    assembly_config_path = "assembly/free_bend_18cm.json"

    # Set New Configuration
    original_database_path = "rod_library/standard_18.json"
    with open(original_database_path, "r") as f:
        json_data = f.read()
    rod_library = RodLibrary.model_validate(from_json(json_data))

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

    return np.array([loss])


def obj_fun(pp):
    """Objective function to be minimized."""
    res = objective(**pp)
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
    opt_id = "optimize_free_bend"
    space = {"moment_scale": [1e-4, 1e-1]}
    objective_names = ["loss"]

    # Create an optimizer
    dmosopt_params = {
        "opt_id": opt_id,
        "obj_fun_name": "optimize_free_bend.obj_fun",
        "obj_fun_init_args": {},
        "problem_parameters": {},
        "space": space,
        "objective_names": objective_names,
        "population_size": 200,
        "num_generations": 200,
        "initial_maxiter": 10,
        "optimizer_name": "nsga2",
        "termination_conditions": True,
        "n_initial": 5,
        "n_epochs": 10,
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
    # besty = best["y"]

    optimizer_data = dmosopt.load_h5_optimizer_data(opt_id=opt_id)
    results = dmosopt.load_h5(opt_id=opt_id)
    x = results["parameters"].values
    y = results["objectives"]["loss"].values

    # plot results
    plt.scatter(x[:, 0], y)
    plt.scatter(bestx["moment_scale"], y.min(), label="best points", marker="x")
    plt.xlabel("Moment Scale")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("optimize_free_bend.svg")
