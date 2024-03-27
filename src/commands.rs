// QEIR, simulator of a monolayer of cells that hold a simple internal state and communicate when in contact
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use std::fs::File;
use crate::compartment::Compartment;
use crate::lattice::Lattice;
use crate::output::Output;
use crate::parameters::Parameters;
use crate::simulation::Simulation;

use rand::rngs::StdRng;

pub fn initialize_front(lattice: &mut Lattice) {
    for y in 0..lattice.height {
        // barrier
        let ref mut c = lattice.cells[0 * lattice.height + y];
        c.alive = false;

        // I
        let ref mut c = lattice.cells[1 * lattice.height + y];
        c.compartments[Compartment::I as usize] = 1;
        c.compartments[Compartment::E as usize] = 0;
        c.compartments[Compartment::R as usize] = 0;
    }
}

pub fn run_simulation(
    lattice: &mut Lattice,
    parameters: &Parameters,
    rng: &mut StdRng,
    tspan: (f64, f64),
    files_out_interval: f64,
    maybe_output: &Option<Output>,
    out_init_frame: bool,
    maybe_activity_horizontal_csv: &Option<File>
) {
    Simulation::new(lattice).run(
        parameters,
        rng,
        tspan,
        maybe_output,
        files_out_interval,
        out_init_frame,
        maybe_activity_horizontal_csv
    );
}