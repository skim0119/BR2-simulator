# Setup Simulation

To run the simulation, you need to provide `database` and `assembly` script in `json` format.
User can register their available _FREE_ specs in the database file, and assembly script can include the registered _FREE_ to assemble the arm.

The activation of the arm can be encoded within the assembly script. In the simulation environment, user can provide the activation in dictionary (i.e. `action={"bend-group1":40, "twist-group2":15}`).

The example of the data-base is included [here](https://github.com/skim0119/BR2-simulator/tree/main/sample_database).
The example of assembly script is included [here](https://github.com/skim0119/BR2-simulator/tree/main/sample_assembly).

:::{Note}
At the current stage, we only support simple serial and parallel assembly of the FREE to build an arm. The purpose is to explore the potential of soft-arm within the ability of fabrication and manufacturing. If you want to explore customized types of connections and constraint, you can try to directly implement using [PyElastica](https://docs.cosseratrods.org/en/latest/).
:::

## How to write Database in Json

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

## How to write Free Assembly in Json

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
