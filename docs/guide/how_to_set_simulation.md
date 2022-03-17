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

## How to write Free Assembly in Json
