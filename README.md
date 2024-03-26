QEIR
====
A simulator of a confluent monolayer of cells that hold a simple internal
state and may communicate when in contact.

Allowed cell states and transitions between them are patterned after
a multi-compartment epidemiological SEIR model (wherein S -- susceptible,
E -- exposed, I -- infectious, R -- resistant/recovered).  As the simulator
is used primarily to investigate spatial stochastic kinetics of waves of
activation, the 'susceptible' compartment has been renamed to the 'quiescent'
compartment, which is more appropriate in the cellular context.

The following transitions are assumed:
```
 Q -> E_1 -> ... -> E_{nE} -> I_1 -> ... -> I_{nI} -> R_1 -> ... -> R_{nR} -> Q
---  `----- exposed -----'    `--- infectious ---'    `--- recovering ---'   ---

```

Model definition
----------------

The model is parametrized by specifying the numbers of the E, I, and R
subcompartments and four kinetic rate constants for (forward-only) transitions.
The respective **parameters** are
`e_subcompartments_count`,
`i_subcompartments_count`,
`r_subcompartments_count`,
`c_rate`,
`e_forward_rate`,
`i_forward_rate`,
`r_forward_rate`.

The simulation **protocol** is specified by a sequence of the two commands:
* `[]!` -- triggers a new front,
* `run` -- runs a simulation in a specified time span with the output time interval specified in square brackets (e.g., `run 0h...30m [5s]`).


Running
-------
The simulator is implemented in Rust.  To compile and run the executable
binary, in terminal type:
````
cargo run --release -- --height 7 --width 100 --states-out --activity-out --images-out input/parameters.json input/one_initial_front.proto
````
For an explanation of the command-line arguments, type:
````
cargo run --release -- --help
````


Origin
------
QEIR is a fork and simplification of our viral infection simulator, VIS-A-VIS
(https://github.com/grfrederic/visavis).
