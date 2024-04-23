QEIR
====
A simulator of a confluent monolayer of cells that hold a simple internal
state and may communicate when in contact.

Model
-----
Cell states patterned after a multi-compartment epidemiological SEIR model
were renamed to match the cellular signaling context: 
* Q - quiescent,
* E - excited,
* I - inducing
* R - refracory.

The following **transitions** are assumed:
```
 Q -> E_1 -> ... -> E_{nE} -> I_1 -> ... -> I_{nI} -> R_1 -> ... -> R_{nR} -> Q
      ------"Excited"-----    -----"Inducing"-----    ----"Refractory"----
```

The transition Q -> E_1 is induced by a neighboring cell in one of 
the "Inducing" states.

The model is parametrized by specifying the numbers of the E, I, and R
subcompartments and four kinetic rate constants for (forward-only) transitions.
The respective **parameters** are:
* `e_subcompartments_count` (`nE` in the scheme above),
* `i_subcompartments_count` (`nI` in the scheme above),
* `r_subcompartments_count` (`nR` in the scheme above),
* `c_rate` (receiving activation from a Q neighbor to become E_1),
* `e_forward_rate` (progression within and out of the "Excited" states),
* `i_forward_rate` (progression within and out of the "Inducing" states),
* `r_forward_rate` (progression within and out of the "Refractory" states).

The simulation **protocol** is specified by a sequence of the two commands:
* `+front at column 0` -- triggers a new front of activity by turning cells in column 0 to state I_1,
* `run` -- runs a simulation in a specified time span with the output time
interval specified in square brackets (e.g., `run 0h...30m [5s]`); when the
output time interval is not specified (as in, e.g., `run 0h...30m []`), then
no output files are produced.

The reactor geometry is set upon launching the simulator with command-line
arguments.


Running
-------
The simulator is implemented in Rust. To compile and run the executable
binary, in terminal type:
````
cargo run --release -- --height 7 --width 100 --states-out --activity-out --images-out input/parameters.json input/one_initial_front.proto
````
For an explanation of the command-line arguments, type:
````
cargo run --release -- --help
````
If libglib2.0-dev and libcairo2-dev are installed, all dependencies should
be retrieved and compiled on the fly prior to simulator compilation and execution.


Origin
------
QEIR is a fork and simplification of our viral infection simulator, VIS-A-VIS
(https://github.com/grfrederic/visavis).
