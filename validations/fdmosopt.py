from typing import Dict, Optional, List, Callable, Literal, Set, Any, Union, Tuple
from mpi4py import MPI
import os
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    TypeAdapter,
)
from dmosopt import dmosopt
from dmosopt import config as dmosopt_config_tools
from numbers import Number
import copy
import sys
import inspect
import h5py
from dmosopt.MOASMO import get_best
from models.utils import epsilon_get_best
from dmosopt import indicators
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys.excepthook = mpi_excepthook


class DmosoptConfig(BaseModel):
    #model_config = ConfigDict(extra="forbid")
    model_config = ConfigDict()

    obj_fun: Callable | None = None
    controller: Callable | Any | None = None
    reduce_fun: Callable | None = None

    # Basic configuration
    opt_id: str
    obj_fun_name: str | None
    obj_fun_init_name: str | None = None
    obj_fun_init_args: dict[str, Any] = {}
    controller_init_fun_name: str | None = None
    controller_init_fun_args: dict[str, Any] | None = None
    reduce_fun_name: str | None = None
    reduce_fun_args: Union[List, Tuple] | None = None
    broker_fun_name: Optional[str] = None
    broker_module_name: Optional[str] = None
    # DistOptimizer
    objective_names: list[str] = []
    feature_dtypes: str | None = None
    constraint_names: Union[str, List[str]] = None
    n_initial: int
    initial_maxiter: int
    initial_method: Union[
        Callable,
        Literal["glp", "slh", "lh", "mc", "sobol"],
        Dict[str, Any],
        str,
    ] = "glp"
    dynamic_initial_sampling: str | None = None
    dynamic_initial_sampling_kwargs: dict[str, Any] = {}
    problem_ids: Set | None = None
    problem_parameters: Optional[Dict]
    space: dict[str, tuple[int, int] | tuple[float, float]] = {}
    population_size: int
    num_generations: int
    resample_fraction: float = 0.25
    distance_metric: Union[Callable, Literal["crowding", "euclidean"]] | None = None
    n_epochs: int
    save_eval: bool = True
    file_path: str
    save: bool = True
    save_surrogate_evals: bool = True
    save_optimizer_params: bool = True
    metadata: Any | None = None
    surrogate_method_name: Literal[
        "gpr",
        "egp",
        "megp",
        "mdgp",
        "mdspp",
        "vgp",
        "svgp",
        "spv",
        "siv",
        "crv",
    ] = "gpr"
    surrogate_method_kwargs: dict[str, Any] = {
        "anisotropic": False,
        "optimizer": "sceua",
    }
    surrogate_custom_training: str | None = None
    surrogate_custom_training_kwargs: dict[str, Any] = {}
    optimizer_name: Literal["nsga2", "age", "smpso", "cmaes"] = "nsga2"
    optimizer_kwargs: dict[str, Any] = {"crossover_prob": 0.9, "mutation_prob": 0.1}
    sensitivity_method_name: Literal["dgsm", "fast"] | None = None
    sensitivity_method_kwargs: dict[str, Any] = {}
    random_seed: int = 0
    feasibility_method_name: str | None = None
    feasibility_method_kwargs: dict[str, Any] = {}
    termination_conditions: Union[bool, Dict] | None = True
    #
    di_crossover: Any | None = None
    di_mutation: Any | None = None

    @model_validator(mode="before")
    @classmethod
    def valid_optimization_settings(cls, payload: Any) -> Any:
        # validate imports eagerly
        for path, alias, kw in [
            ("obj_fun_name", {}, None),
            ("obj_fun_init_name", {}, "obj_fun_init_args"),
            ("controller_init_fun_name", {}, "controller_init_fun_args"),
            ("reduce_fun_name", {}, None),
            ("broker_fun_name", {}, None),
            ("initial_method", dmosopt_config_tools.default_sampling_methods, None),
            ("dynamic_initial_sampling", {}, "dynamic_initial_sampling_kwargs"),
            (
                "surrogate_method_name",
                dmosopt_config_tools.default_surrogate_methods,
                "surrogate_method_kwargs",
            ),
            (
                "feasibility_method_name",
                dmosopt_config_tools.default_feasibility_methods,
                "feasibility_method_kwargs",
            ),
            ("surrogate_custom_training", {}, "surrogate_custom_training_kwargs"),
            ("optimizer_name", dmosopt_config_tools.default_optimizers, None),
            (
                "sensitivity_method_name",
                dmosopt_config_tools.default_sa_methods,
                "sensitivity_method_kwargs",
            ),
            ("feature_dtypes", {}, None),
            ("objective_names", {}, None),
            ("constraint_names", {}, None),
            ("metadata", {}, None),
        ]:
            target = payload.get(path, None)
            if not isinstance(target, str):
                continue
            if target in alias:
                target = alias[target]

            try:
                obj = dmosopt_config_tools.import_object_by_path(target)
            except ImportError as _ex:
                raise ValueError(
                    f"Could not resolve import path '{target}' for '{path}': {_ex}"
                ) from _ex

            if kw is None:
                continue
            d = payload[kw]
            if not d:
                continue

            # verify arguments
            sig = inspect.signature(obj)
            if any([p.kind == p.VAR_KEYWORD for p in sig.parameters.values()]):
                # if function accepts keyword arguments, we cannot validate :-(
                continue

            for key in d.keys():
                if key in sig.parameters:
                    continue
                message = ""
                for name, param in sig.parameters.items():
                    if param.default is param.empty:
                        message += f"{name}, "
                    else:
                        message += f"{name}={param.default}, "
                raise ValueError(
                    f"Invalid {kw} for {target}. Found `{key}`, but signature is {message[:-2]}"
                )

        return payload


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_limit: Optional[int] = None
    feasible: bool = True
    return_features: bool = False
    return_constraints: bool = False
    spawn_workers: bool = False
    sequential_spawn: bool = False
    spawn_startup_wait: Optional[int] = None
    spawn_executable: Optional[str] = None
    spawn_args: List[str] = []
    nprocs_per_worker: int = 1
    collective_mode: Literal["gather", "sendrecv"] = "gather"
    verbose: bool = True
    worker_debug: bool = False
    nodes: str = "20"
    ranks: Optional[int] = None


class Dmosopt(BaseModel):
    dmosopt_config: DmosoptConfig
    run_config: RunConfig = RunConfig()

    def __call__(self) -> None:
        run = dmosopt.run(
            dopt_params=self.dmosopt_config.dict(),
            time_limit=self.run_config.time_limit,
            feasible=self.run_config.feasible,
            return_features=self.run_config.return_features,
            return_constraints=self.run_config.return_constraints,
            spawn_workers=self.run_config.spawn_workers,
            sequential_spawn=self.run_config.sequential_spawn,
            spawn_startup_wait=self.run_config.spawn_startup_wait,
            spawn_executable=self.run_config.spawn_executable,
            spawn_args=self.run_config.spawn_args,
            nprocs_per_worker=self.run_config.nprocs_per_worker,
            collective_mode=self.run_config.collective_mode,
            #verbose=self.run_config.verbose,
            worker_debug=self.run_config.worker_debug,
        )

    def parameter_vector_to_dict(self, x, include_constants=True):
        constants = {}
        if include_constants:
            constants = self.dmosopt_config.problem_parameters
        return {
            **constants,
            **{k: x[n] for n, k in enumerate(self.dmosopt_config.space.keys())},
        }

    def evaluate_objective_at(self, x):
        import logging

        p = x
        if not isinstance(p, dict):
            p = self.parameter_vector_to_dict(x)

        logging.basicConfig(level=logging.INFO)
        if "obj_fun_init_name" in self.dmosopt_config:
            obj_fun = dmosopt_config_tools.import_object_by_path(
                self.dmosopt_config.obj_fun_init_name
            )(**self.dmosopt_config.obj_fun_init_args)
        else:
            obj_fun = dmosopt_config_tools.import_object_by_path(
                self.dmosopt_config.obj_fun_name
            )

        return obj_fun(p)

    @property
    def output_filepath(self) -> str:
        return os.path.abspath(self.dmosopt_config.file_path)

    def on_write_meta_data(self):
        return MPI.COMM_WORLD.Get_rank() == 0

    def on_commit(self):
        if MPI.COMM_WORLD.Get_rank() > 0:
            return False

    def load_h5(
        self,
        opt_id: Optional[str] = None,
        problem_id: int = 0,
    ):
        filepath = self.output_filepath

        if opt_id is None:
            opt_id = self.dmosopt_config.opt_id

        with h5py.File(filepath, "r") as h5:
            # constraints
            if f"{opt_id}/constraint_enum" in h5:
                constraint_enum = h5py.check_enum_dtype(
                    h5[f"{opt_id}/constraint_enum"].dtype
                )
                constraint_enum_T = {v: k for k, v in constraint_enum.items()}
                constraint_names = [
                    constraint_enum_T[s[0]]
                    for s in iter(h5[f"{opt_id}/constraint_spec"])
                ]
                constraints = pd.DataFrame(
                    h5[f"{opt_id}/{problem_id}/constraints"][:],
                    columns=constraint_names,
                )
                if self.constraint_names:
                    constraints = constraints[
                        self.constraint_names
                    ]  # sort for consistency
            else:
                constraints = None

            # epochs
            epochs = h5[f"{opt_id}/{problem_id}/epochs"][:]

            # features
            if f"{opt_id}/feature_enum" in h5:
                feature_enum = h5py.check_enum_dtype(h5[f"{opt_id}/feature_enum"].dtype)
                feature_enum_T = {v: k for k, v in feature_enum.items()}
                feature_names = [
                    feature_enum_T[s[0]] for s in iter(h5[f"{opt_id}/feature_spec"])
                ]
                features = pd.DataFrame(
                    [
                        list(feature)
                        for feature in h5[f"{opt_id}/{problem_id}/features"]
                    ],
                    columns=feature_names,
                )
                if self.feature_names:
                    features = features[self.feature_names]  # sort for consistency
            else:
                features = None

            # objectives
            objective_enum = h5py.check_enum_dtype(h5[f"{opt_id}/objective_enum"].dtype)
            objective_enum_T = {v: k for k, v in objective_enum.items()}
            objective_names = [
                objective_enum_T[s[0]] for s in iter(h5[f"{opt_id}/objective_spec"])
            ]
            objectives = pd.DataFrame(
                h5[f"{opt_id}/{problem_id}/objectives"][:], columns=objective_names
            )
            if self.objective_names:
                objectives = objectives[self.objective_names]  # sort for consistency

            # parameters
            parameter_enum = h5py.check_enum_dtype(h5[f"{opt_id}/parameter_enum"].dtype)
            parameter_enum_T = {v: k for k, v in parameter_enum.items()}
            parameter_names = [
                parameter_enum_T[s[0]] for s in iter(h5[f"{opt_id}/parameter_spec"])
            ]
            parameters = pd.DataFrame(
                h5[f"{opt_id}/{problem_id}/parameters"][:], columns=parameter_names
            )
            # order such that it stays consistent with the space definition
            parameters = parameters[list(self.dmosopt_config.space.keys())]

            # predictions
            predictions = pd.DataFrame(
                h5[f"{opt_id}/{problem_id}/predictions"][:], columns=objective_names
            )
            if self.objective_names:
                predictions = predictions[self.objective_names]  # sort for consistency

            # metadata
            metadata = None
            if f"/{opt_id}/metadata" in h5:
                metadata = h5[f"/{opt_id}/metadata"][:]

        return {
            "constraints": constraints,
            "epochs": epochs,
            "features": features,
            "objectives": objectives,
            "parameters": parameters,
            "predictions": predictions,
            "metadata": metadata,
        }

    def load_h5_optimizer_data(self, opt_id: Optional[str] = None):
        filepath = self.output_filepath

        if opt_id is None:
            opt_id = self.dmosopt_config.opt_id

        with h5py.File(filepath, "r") as h5:
            stats = None
            if f"/{opt_id}/optimizer_stats" in h5:
                epoch = 0
                stats = []
                while True:
                    if f"/{opt_id}/optimizer_stats/{epoch}" not in h5:
                        break

                    epoch_stats = h5[f"/{opt_id}/optimizer_stats/{epoch}/stats"]
                    stats.append(epoch_stats[:])

                    epoch += 1

                items = [np.array(stat.item()) for stat in stats]
                stats = pd.DataFrame(
                    items, columns=epoch_stats.dtype.names
                )

            params = None
            if f"/{opt_id}/optimizer_params" in h5:
                epoch = 1
                params = []
                while True:
                    if f"/{opt_id}/optimizer_params/{epoch}" not in h5:
                        break

                    epoch_params = h5[f"/{opt_id}/optimizer_params/{epoch}"]
                    row = {"epoch": epoch}
                    for dset in epoch_params:
                        row[dset] = epoch_params[dset][()]
                    params.append(row)

                    epoch += 1

                params = pd.DataFrame(params)

        return {"stats": stats, "params": params}

    def load_h5_surrogate_evals(
        self,
        opt_id: Optional[str] = None,
        problem_id: int = 0,
    ):
        filepath = self.output_filepath

        if opt_id is None:
            opt_id = self.dmosopt_config.opt_id

        with h5py.File(filepath, "r") as h5:
            epochs = None
            if f"/{opt_id}/surrogate_evals/epochs" in h5:
                epochs = h5[f"/{opt_id}/surrogate_evals/epochs"][:]

            generations = None
            if f"/{opt_id}/surrogate_evals/generations" in h5:
                generations = h5[f"/{opt_id}/surrogate_evals/generations"][:]

            objectives = None
            if f"/{opt_id}/surrogate_evals/objectives" in h5:
                objective_enum = h5py.check_enum_dtype(
                    h5[f"{opt_id}/objective_enum"].dtype
                )
                objective_enum_T = {v: k for k, v in objective_enum.items()}
                objective_names = [
                    objective_enum_T[s[0]] for s in iter(h5[f"{opt_id}/objective_spec"])
                ]
                objectives = pd.DataFrame(
                    h5[f"/{opt_id}/surrogate_evals/objectives"][:],
                    columns=objective_names,
                )

            parameters = None
            if f"/{opt_id}/surrogate_evals/parameters" in h5:
                parameter_enum = h5py.check_enum_dtype(
                    h5[f"{opt_id}/parameter_enum"].dtype
                )
                parameter_enum_T = {v: k for k, v in parameter_enum.items()}
                parameter_names = [
                    parameter_enum_T[s[0]] for s in iter(h5[f"{opt_id}/parameter_spec"])
                ]
                parameters = pd.DataFrame(
                    h5[f"/{opt_id}/surrogate_evals/parameters"][:],
                    columns=parameter_names,
                )

        return {
            "epochs": epochs,
            "generations": generations,
            "objectives": objectives,
            "parameters": parameters,
        }

    def infer_num_initial_samples(self, problem_id: int = 0) -> int:
        with h5py.File(self.output_filepath, "r") as h5:
            epochs = h5[f"{self.dmosopt_config.opt_id}/{problem_id}/epochs"][:]

        self.inferred_num_initial_samples = len(epochs[epochs == 0])

        return self.inferred_num_initial_samples

    def get_best(
        self,
        region: list | tuple | None = None,
        sort_by: str = "-np.std(y, axis=1)",
        as_dataframes: bool = True,
        epsilon="auto",
    ):
        data = self.load_h5()

        if region is None:
            if len(data["epochs"]) > 10000:
                # optimize for speed since best solutions will be found in the last epochs
                region = slice(-10000, None)
            else:
                region = slice(None)
        else:
            region = slice(*region)

        objectives = data["objectives"].to_numpy()[region]

        valid = np.logical_not(np.any(np.isnan(objectives), axis=1))

        y = objectives[valid]
        x = data["parameters"].to_numpy()[region][valid]
        if data["constraints"] is not None:
            C = data["constraints"].to_numpy()[region][valid]
        else:
            C = None
        if data["features"] is not None:
            f = data["features"].to_numpy()[region][valid]
        else:
            f = None

        if epsilon is not None or len(x) == 0:
            best_x, best_y, best_f, best_c, eps = epsilon_get_best(
                x, y, f, C, epsilons=epsilon
            )
        else:
            # strict non-dominated sort
            best_x, best_y, best_f, best_c, best_epoch, perm = get_best(
                x, y, f, C, None, None
            )

        if isinstance(sort_by, str):
            if len(best_x) > 0:
                context = {
                    "reduced": None,
                    "x": best_x,
                    "y": best_y,
                    "f": best_f,
                    "c": best_c,
                    "np": np,
                }
                exec(f"reduced={sort_by}", context)
                sort_by = np.argsort(context["reduced"])
            else:
                sort_by = None

        best = {"x": best_x, "y": best_y, "f": best_f, "c": best_c}

        # apply sort
        if sort_by is not None:
            for k in best.keys():
                if best[k] is not None:
                    best[k] = best[k][sort_by]

        if as_dataframes:
            best["x"] = pd.DataFrame(best["x"], columns=data["parameters"].columns)
            best["y"] = pd.DataFrame(best["y"], columns=data["objectives"].columns)
            if best["f"] is not None:
                best["f"] = pd.DataFrame(best["f"], columns=data["features"].columns)
            if best["c"] is not None:
                best["c"] = pd.DataFrame(best["c"], columns=data["constraints"].columns)

        return best

    def front(self, pf=None):
        if pf is None:
            return self.get_best()["y"].to_numpy()
        elif isinstance(pf, pd.DataFrame):
            return pf.to_numpy()
        elif isinstance(pf, pd.Series):
            return pf.to_numpy()
        elif isinstance(pf, Dmosopt):
            return pf.get_best()["y"].to_numpy()
        else:
            return np.array(pf)

    def norm_front(self, pf, min_max=None):
        pf = self.front(pf)

        if not isinstance(min_max, (list, tuple)):
            fmin, fmax = np.min(pf, axis=0), np.max(pf, axis=1)
        else:
            fmin, fmax = np.array(min_max[0]), np.array(min_max[1])

        return (pf - fmin) / (fmax - fmin)

    def igd(self, ref_front, pf=None):
        ref_front, pf = self.front(ref_front), self.front(pf)

        indicator = indicators.IGD(np.array(pf))

        return indicator.do(np.array(ref_front))

    def hypervolume(self, ref_point, pf=None, normalize=False):
        if normalize:
            pf = self.norm_front(pf, normalize)
        else:
            pf = self.front(pf)

        indicator = indicators.Hypervolume(np.array(ref_point))

        return indicator.do(np.array(pf))

    def c_metric(self, ref_front, pf=None):
        """
        Calculates the set coverage of A over B, i.e. C(A, B),
        which is the fraction of solutions in B that are
        dominated by at least one solution in A.

        ref_front: B front array
        pf: A front array
        """
        ref_front, pf = self.front(ref_front), self.front(pf)

        coverage_count = 0
        for candidate in ref_front:
            for solution in pf:
                # solution dominates candidate?
                if all(r <= c for r, c in zip(solution, candidate)) and any(
                    r < c for r, c in zip(solution, candidate)
                ):
                    coverage_count += 1
                    break
        return coverage_count / len(ref_front)

    @property
    def dc(self):
        return self.dmosopt_config

    @property
    def constraint_names(self) -> list[str]:
        cn = self.dmosopt_config.get("constraint_names", [])
        if isinstance(cn, str):
            cn = dmosopt_config_tools.import_object_by_path(cn)
            if callable(cn):
                cn = cn(self)
        return cn

    @property
    def num_constraints(self) -> int:
        return len(self.constraint_names)

    @property
    def objective_names(self) -> list[str]:
        on = self.dmosopt_config.objective_names
        if isinstance(on, str):
            on = dmosopt_config_tools.import_object_by_path(on)
            if callable(on):
                on = on(self)
        return on

    @property
    def num_objectives(self) -> int:
        return len(self.objective_names)

    @property
    def feature_names(self) -> list[str]:
        return [f[0] for f in self.feature_dtypes]

    @property
    def feature_dtypes(self) -> list[str]:
        fn = self.dmosopt_config.get("feature_dtypes", [])
        if isinstance(fn, str):
            fn = dmosopt_config_tools.import_object_by_path(fn)
            if callable(fn):
                fn = fn(self)
        return fn

    @property
    def resample_fraction(self) -> float:
        return self.dmosopt_config.get("resample_fraction", 0.25)

    @property
    def population_size(self) -> int:
        return self.dmosopt_config.get("population_size", 100)

    @property
    def surrogate_method_name(self) -> str:
        return self.dmosopt_config.get("surrogate_method_name", "gpr")

    @property
    def initial_method(self) -> str:
        return self.dmosopt_config.get("initial_method", "slh")

    @property
    def num_generations(self) -> int:
        return self.dmosopt_config.get("num_generations", 200)

    @property
    def n_epochs(self) -> int:
        return self.dmosopt_config.get("n_epochs", 10)

    @property
    def n_initial(self) -> int:
        return self.dmosopt_config.get("n_initial", 10)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    @property
    def num_parameters(self) -> int:
        return len(self.space)

    @property
    def num_initial_samples(self) -> int:
        if self.dmosopt_config.get("dynamic_initial_sampling", None) is not None:
            n_initial = getattr(self, "inferred_num_initial_samples", None)
            if n_initial is None:
                raise RuntimeError(
                    "Dynamic initial sampling is used, so the number of initial samples is not known. Call infer_num_initial_samples() first."
                )
            else:
                return n_initial

        return self.n_initial * self.num_parameters

    @property
    def num_resample(self) -> int:
        return int(self.resample_fraction * self.population_size)

    @property
    def num_evals_per_epoch(self) -> int:
        if (
            self.surrogate_method_name is None
            and self.dmosopt_config.get("surrogate_custom_training", None) is None
        ):
            return self.population_size * self.num_generations + self.num_resample

        return self.num_resample

    @property
    def num_evals_total(self) -> int:
        # n_epochs - 1 since epoch 0 is using the initial sampling, so there are no additional evals
        return self.num_initial_samples + (self.n_epochs - 1) * self.num_evals_per_epoch

    @property
    def num_max_surrogate_evals(self) -> int:
        if (
            self.surrogate_method_name is None
            and self.dmosopt_config.get("surrogate_custom_training", None) is None
        ):
            return 0

        evals = 0
        for epoch in range(1, self.n_epochs - 1):
            # initial sampling
            evals += self.num_initial_samples
            evals += self.population_size * epoch
            # generation
            evals += self.population_size * (self.num_generations + 1)

        return evals

    @property
    def space(self) -> dict[str, Tuple[Number, Number]]:
        return self.dmosopt_config["space"]

    def estimate_run_time(self, eval_seconds, surrogate_eval_seconds=None):
        seconds = self.num_evals_total * eval_seconds
        if surrogate_eval_seconds is not None:
            seconds += self.num_max_surrogate_evals * surrogate_eval_seconds
        return datetime.timedelta(seconds=seconds)

    def h5_config_consistency(self) -> list[tuple[str, Number, Number]]:
        inconsistencies = []

        data = self.load_h5()

        # num_features
        if self.num_features == 0:
            if data["features"] is not None:
                inconsistencies.append(
                    ("num_features", self.num_features, data["features"].shape[1])
                )
        elif self.num_features != data["features"].shape[1]:
            inconsistencies.append(
                ("num_features", self.num_features, data["features"].shape[1])
            )
            if self.feature_names != data["features"].columns.tolist():
                inconsistencies.append(
                    (
                        "feature_names",
                        self.feature_names,
                        data["features"].columns.tolist(),
                    )
                )

        # num_constraints
        if self.num_constraints == 0:
            if data["constraints"] is not None:
                inconsistencies.append(
                    (
                        "num_constraints",
                        self.num_constraints,
                        data["constraints"].shape[1],
                    )
                )
        elif self.num_constraints != data["constraints"].shape[1]:
            inconsistencies.append(
                ("num_constraints", self.num_constraints, data["constraints"].shape[1])
            )
            if self.constraint_names != data["constraints"].columns.tolist():
                inconsistencies.append(
                    (
                        "constraint_names",
                        self.constraint_names,
                        data["constraints"].columns.tolist(),
                    )
                )

        # num_parameters
        if self.num_parameters != data["parameters"].shape[1]:
            inconsistencies.append(
                ("num_parameters", self.num_parameters, data["parameters"].shape[1])
            )
            if self.dmosopt_config.space.keys() != data["parameters"].columns.tolist():
                inconsistencies.append(
                    (
                        "parameter_names",
                        self.dmosopt_config.space.keys(),
                        data["parameters"].columns.tolist(),
                    )
                )

        # num_evals_total
        if self.num_evals_total != len(data["epochs"]):
            inconsistencies.append(
                ("num_evals_total", self.num_evals_total, len(data["epochs"]))
            )

        return inconsistencies
