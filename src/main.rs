// -------------------------------------------------------------------------------------------------
// VIS-A-VIS, a simulator of Viral Infection Spread And Viral Infection Self-containment.
//
// This code features the research article:
//
//       "Antagonism between viral infection and innate immunity at the single-cell level"
//
//                               by Frederic Grabowski et al.
//                                [TODO:JOURNAL-NAME], 202X
//
// The simulation mimicks the innate immune response to an infection with an RNA virus.
// The hard-coded and externally parametrized interactions between host cell and virus
// are specific to the respiratory syncytial virus (RSV). Infected cells attempt to produce
// and secrete interferon, which alerts the non-infected bystander cells about the nearby
// threat. The simulator executes alternating phases of (deterministic) interferon diffusion
// and (stochastic) chemical kinetics.
//
// For more info, see file ReadMe.md.
//
// Copyright (2022) Marek Kochanczyk & Frederic Grabowski (IPPT PAN, Warsaw).
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).
// -------------------------------------------------------------------------------------------------

mod cell;
mod commands;
mod config;
mod event;
mod lattice;
mod molecule;
mod protocol;
mod randomness;
mod rates;
mod simulation;
mod units;

use config::THREAD_STACK_SIZE;
use lattice::Lattice;
use protocol::Protocol;
use randomness::initialize_generator;
use rates::Rates;

use clap::Parser;


#[derive(Parser)]
#[command(version, about)]
struct Args {
    parameters_json_file: String,
    protocol_file: String,
    /// Generate png images for every frame
    #[clap(long = "images", short = 'I', action)]
    images: bool,
    /// Write number of active (E/I) cells in every row for every frame
    #[clap(long = "activity", short = 'A', action)]
    activity: bool,
    /// Write full state for every frame
    #[clap(long = "states", short = 'S', action)]
    states: bool,
    /// Random seed
    #[clap(long = "seed", short = 's', default_value_t = 123)]
    seed: u128,
}


fn execute_protocol() -> bool {
    
    let args = Args::parse();
    let rates = Rates::from_json_file(&args.parameters_json_file);
    let protocol = Protocol::from_text_file(&args.protocol_file);
    let images_out = args.images;
    let activity_out = args.activity;
    let states_out = args.states;
    let seed = args.seed;

    std::thread::Builder::new()
        .name("protocol_execution".into())
        .stack_size(THREAD_STACK_SIZE)
        .spawn(move || {
            let mut generator = initialize_generator(seed, false);
            let mut lattice = Lattice::new(&mut generator);
            protocol.execute(&mut lattice, &rates, &mut generator, images_out, activity_out, states_out);
        })
        .expect("☠ @ protocol_execution thread")
        .join()
        .expect("☠ @ threads join");
    true
}

fn main() {
    let _ = execute_protocol();
}
