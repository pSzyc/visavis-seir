// QEIR, simulator of a monolayer of cells that hold a simple internal state and communicate when in contact
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

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

fn run_simulation_(
    lattice: &mut Lattice,
    parameters: &Parameters,
    rng: &mut StdRng,
    tspan: (f64, f64),
    files_out: bool,
    output: Output,
    files_out_interval: f64,
    out_init_frame: bool,
) {
    let workers = Some(
        threadpool::Builder::new()
            .num_threads(num_cpus::get())
            .build(),
    );

    Simulation::new(lattice).run(
        parameters,
        rng,
        tspan,
        files_out,
        output,
        files_out_interval,
        out_init_frame,
        &workers,
    );
    workers.unwrap().join()
}

pub fn run_simulation_quietly(
    lattice: &mut Lattice,
    parameters: &Parameters,
    rng: &mut StdRng,
    tspan: (f64, f64),
    output: Output,
    out_init_frame: bool,
) {
    run_simulation_(
        lattice,
        parameters,
        rng,
        tspan,
        /*files_out:*/ false,
        output,
        /*files_out_interval*/ -1.,
        out_init_frame,
    )
}

pub fn run_simulation(
    lattice: &mut Lattice,
    parameters: &Parameters,
    rng: &mut StdRng,
    tspan: (f64, f64),
    files_out_interval: f64,
    output: Output,
    out_init_frame: bool,
) {
    run_simulation_(
        lattice,
        parameters,
        rng,
        tspan,
        /*files_out:*/ true,
        output,
        files_out_interval,
        out_init_frame,
    )
}
