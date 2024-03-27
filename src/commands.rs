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

pub fn initialize_front(lattice: &mut Lattice, column: usize) {
    for y in 0..lattice.height {
        {
            // barrier
            let cell = &mut lattice.cells[column * lattice.height + y];
            cell.alive = false;
        }{
            // active ~ "infectious"
            debug_assert!(column + 1 < lattice.width);
            let cell = &mut lattice.cells[(column + 1) * lattice.height + y];
            cell.compartments[Compartment::I as usize] = 1;
            cell.compartments[Compartment::E as usize] = 0;
            cell.compartments[Compartment::R as usize] = 0;
        }
    }
}

pub fn run_simulation(
    lattice: &mut Lattice,
    parameters: &Parameters,
    rng: &mut StdRng,
    tspan: (f64, f64),
    maybe_output: &Option<Output>,
    maybe_activity_horizontal_csv: &Option<File>,
    files_out_interval: f64,
    initial_frame_in_output_files: bool
) {
    Simulation::new(lattice).run(
        parameters,
        rng,
        tspan,
        maybe_output,
        maybe_activity_horizontal_csv,
        files_out_interval,
        initial_frame_in_output_files,
    );
}