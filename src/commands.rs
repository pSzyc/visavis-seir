// VIS-A-VIS, a simulator of Viral Infection Spread And Viral Infection Self-containment.
//
// Copyright (2022) Marek Kochanczyk & Frederic Grabowski (IPPT PAN, Warsaw).
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use crate::config::THREAD_STACK_SIZE;
use crate::lattice::Lattice;
use crate::molecule::Mol;
use crate::rates::Rates;
use crate::legal_states::LegalStates;
use crate::simulation::Simulation;
use std::fs::{File};

use rand::rngs::StdRng;

pub fn initialize_epidemics(lattice: &mut Lattice) {
    for w in 0..Lattice::WIDTH {

        // barrier
        let ref mut c = lattice.cells[0*Lattice::WIDTH + w];
        c.alive = false;

        // I
        let ref mut c = lattice.cells[1*Lattice::WIDTH + w];
        c.molecules[Mol::I as usize] = 1;
        c.molecules[Mol::E as usize] = 0;
        c.molecules[Mol::R as usize] = 0;
    }
}

pub fn run_simulation_quietly(
    lattice: &mut Lattice,
    rates: &Rates,
    legal_states: &LegalStates,
    rng: &mut StdRng,
    tspan: (f64, f64),
    images_out: bool,
    states_out: bool,
    init_frame_out: bool,
    activity_csv: Option<&File>,
) {
    run_simulation_(
        lattice,
        rates,
        legal_states,
        rng,
        tspan,
        /*files_out:*/ false,
        images_out,
        states_out,
        /*files_out_interval*/ -1.,
        init_frame_out,
        activity_csv,
    )
}

pub fn run_simulation(
    lattice: &mut Lattice,
    rates: &Rates,
    legal_states: &LegalStates,
    rng: &mut StdRng,
    tspan: (f64, f64),
    images_out: bool,
    states_out: bool,
    files_out_interval: f64,
    init_frame_out: bool,
    activity_csv: Option<&File>,
) {
    run_simulation_(
        lattice,
        rates,
        legal_states,
        rng,
        tspan,
        /*files_out:*/ true,
        images_out,
        states_out,
        files_out_interval,
        init_frame_out,
        activity_csv,
    )
}

fn run_simulation_(
    lattice: &mut Lattice,
    rates: &Rates,
    legal_states: &LegalStates,
    rng: &mut StdRng,
    tspan: (f64, f64),
    files_out: bool,
    images_out: bool,
    states_out: bool,
    files_out_interval: f64,
    init_frame_out: bool,
    activity_csv: Option<&File>,
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
        legal_states,
        rng,
        tspan,
        files_out,
        images_out,
        states_out,
        files_out_interval,
        /*ifni_secretion:*/ true,
        /*in_sep_thread:*/ false,
        init_frame_out,
        activity_csv,
        &workers,
    );
    workers.unwrap().join()
}
