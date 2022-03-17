<div align="center">
<h1> BR2 Simulator </h1>

<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>
<a href='https://br2-simulator.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/br2-simulator/badge/?version=latest' alt='Documentation Status'/>
</a>

</div>

----

Simulator development for BR2 Softrobot using [`PyElastica`](https://github.com/GazzolaLab/PyElastica).

The [documentation](https://br2-simulator.readthedocs.io/en/latest/index.html) includes the detail guide of how to use this package.

## Configuration

### Rod Library

If the rod material property is not specified in `Rod` class, the default parameter will be used to construct the rod.
The fiber can be added on the rod using `alpha` and `beta` parameter which takes the list of fiber angles in degrees.
It is recommended to use the angle above 60 degrees, because the current actuation actuation model assumes negligible radial displacement and perpendicular cross-section.
To simulate the bending, apply `gamma` angle (in degrees) which is the cosine angle of the direction of bending moment.

```json
{ 
    "Info": "Example rod properties",
    "DefaultParams": {
        "n_elements"     : 41,
        "direction"      : [0.0,1.0,0.0],
        "normal"         : [0.0,0.0,1.0],
        "base_length"    : 0.18,
        "base_radius"    : 0.007522,
        "density"        : 1500,
        "nu"             : 0.089178,
        "youngs_modulus" : 1e7,
        "poisson_ratio"  : 0.5
    },
    "Rods": {
        "SampleRod": {
            "Info": "Simple",
            "alpha"          : [60],
            "beta"          : [60],
            "gamma"          : [120],
            "youngs_modulus" : 1e7,
            "base_length"    : 0.20
        }
    }
}
```

### Assembly Configuration

`Segments` class defines the structure of each segment, and each segment will be serially connected to the tip of the previous segment.
`Activations` can be defined jointly for multiple rod, which resembles shared actuation pressure.
The position is defined in z-x coordinate which locates the base of the rod.

```json
{ 
    "CaseID": 1,
    "Date": "2021-10-11",
    "Info": "Single-segment BR2 Assembly",
    "Segments": {
        "seg1": {
            "rod_order": ["RodBend", "RodRightTwist", "RodLeftTwist"],
            "base_position": [
                [0.0, 0.0],
                [0.015044, 0.0],
                [0.007522, 0.0130285]
            ]
        }
    },
    "Activations": {
        "action1": [
            ["seg1", "RodBend"]
        ],
        "action2": [
            ["seg1", "RodRightTwist"]
        ],
        "action3": [
            ["seg1", "RodLeftTwist"]
        ]
    },
    "Misc": {
        "Gravity": "Off"
    }
}
```

## Procedures

### Surface Connection Test

- run_testcases_surface_connection.py
    - set_environment_testcases.py
    - surface_connection_parallel_rod_numba.py
    - custom_activation.py
    - custom_constraint.py

The script is testing parallel surface connection with various arm constructions.
This procedure includes customized [activation](custom_activation.py) and [constraint](custom_constraint.py) implementation.
The [surface connection implementation](surface_connection_parallel_rod_numba.py) is used throughout the other procedures.
Different test cases are available, and they will be running using multiprocessing.

### Interactive BR2

_work in progress_

- run_br2_interactive.py

Interactive simulation module.

## Miscellaneous 

- skeleton.py

