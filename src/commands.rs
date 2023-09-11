// VIS-A-VIS, a simulator of Viral Infection Spread And Viral Infection Self-containment.
//
// Copyright (2022) Marek Kochanczyk & Frederic Grabowski (IPPT PAN, Warsaw).
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use crate::config::THREAD_STACK_SIZE;
use crate::lattice::Lattice;
use crate::cell::Cell;
use crate::molecule::Mol;
use crate::rates::Rates;
use crate::simulation::Simulation;

use rand::rngs::StdRng;

pub fn initialize_epidemics(lattice: &mut Lattice) {
    for h in 0..Lattice::HEIGHT {

        // barrier
        let ref mut c = lattice.cells[h*Lattice::WIDTH + 0];
        c.alive = false;

        // I
        let ref mut c = lattice.cells[h*Lattice::WIDTH + 1];
        c.molecules[Mol::I as usize] = 1;
        c.molecules[Mol::E as usize] = 0;
        c.molecules[Mol::R as usize] = 0;

        /* R
        for w in 10..30 { let ref mut c = lattice.cells[h*Lattice::WIDTH + w];
            c.molecules[Mol::R as usize] = Cell::MAX.molecules[Mol::R as usize];
        }
        for w in 30..50 {
            let ref mut c = lattice.cells[h*Lattice::WIDTH + w];
            c.molecules[Mol::R as usize] = 1;
        }

        // I
        let ref mut c = lattice.cells[h*Lattice::WIDTH + 50];
        c.molecules[Mol::I as usize] = Cell::MAX.molecules[Mol::E as usize];
        let ref mut c = lattice.cells[h*Lattice::WIDTH + 51];
        c.molecules[Mol::I as usize] = 1;

        // E
        let ref mut c = lattice.cells[h*Lattice::WIDTH + 52];
        c.molecules[Mol::E as usize] = Cell::MAX.molecules[Mol::E as usize];
        let ref mut c = lattice.cells[h*Lattice::WIDTH + 53];
        c.molecules[Mol::E as usize] = 1;
        */
    }
}

pub fn run_simulation_quietly(
    lattice: &mut Lattice,
    rates: &Rates,
    rng: &mut StdRng,
    tspan: (f64, f64),
    images_out: bool,
    init_frame_out: bool,
) {
    run_simulation_(
        lattice,
        rates,
        rng,
        tspan,
        /*files_out:*/ false,
        images_out,
        /*files_out_interval*/ -1.,
        init_frame_out,
    )
}

pub fn run_simulation(
    lattice: &mut Lattice,
    rates: &Rates,
    rng: &mut StdRng,
    tspan: (f64, f64),
    images_out: bool,
    files_out_interval: f64,
    init_frame_out: bool,
) {
    run_simulation_(
        lattice,
        rates,
        rng,
        tspan,
        /*files_out:*/ true,
        images_out,
        files_out_interval,
        init_frame_out,
    )
}

fn run_simulation_(
    lattice: &mut Lattice,
    rates: &Rates,
    rng: &mut StdRng,
    tspan: (f64, f64),
    files_out: bool,
    images_out: bool,
    files_out_interval: f64,
    init_frame_out: bool,
) {
    let workers = Some(
        threadpool::Builder::new()
            .num_threads(num_cpus::get())
            .thread_stack_size(THREAD_STACK_SIZE)
            .build(),
    );
    Simulation::simulate(
        lattice,
        rates,
        rng,
        tspan,
        files_out,
        images_out,
        files_out_interval,
        /*ifni_secretion:*/ true,
        /*in_sep_thread:*/ false,
        init_frame_out,
        &workers,
    );
    workers.unwrap().join()
}
