// -------------------------------------------------------------------------------------------------
// QEIR -- a simulator of a monolayer of cells that hold a simple internal state and communicate
// when in contact.
//
// This code features the research article:
//
//               "Information transmission in a cell monolayer: a numerical study"
//
//                                 by Nałęcz-Jawecki et al.
//                                [TODO:JOURNAL-NAME], 202X
//
// For more info, see file README.md.
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).
// -------------------------------------------------------------------------------------------------

mod cell;
mod commands;
mod compartment;
mod event;
mod lattice;
mod output;
mod parameters;
mod protocol;
mod randomness;
mod rates;
mod simulation;
mod subcompartments;
mod units;

use lattice::Lattice;
use output::Output;
use parameters::Parameters;
use protocol::Protocol;
use randomness::initialize_generator;

use clap::Parser;

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// A JSON-formatted file with both the kinetic and the structural model parameters
    parameters_file: String,

    /// A file containing commands that may be used as external event triggers
    protocol_file: String,

    /// Lattice width
    #[clap(long = "width", short = 'W', action)]
    width: usize,

    /// Lattice height
    #[clap(long = "height", short = 'H', action)]
    height: usize,

    /// Write out the full state for every output time point
    #[clap(long = "states-out", short = 'S', action)]
    states: bool,

    /// Write the number of active (E or I) cells in every column for every output time point
    #[clap(long = "activity-out", short = 'A', action)]
    activity: bool,

    /// Generate a PNG image for every output time point
    #[clap(long = "images-out", short = 'I', action)]
    images: bool,

    /// Random seed
    #[clap(long = "seed", short = 's', default_value_t = 123)]
    seed: u128,
}

fn execute_protocol() {
    let args = Args::parse();

    let protocol = Protocol::from_text_file(&args.protocol_file);
    let parameters = Parameters::from_json_file(&args.parameters_file);

    let mut generator = initialize_generator(args.seed, false);
    let mut lattice = Lattice::new(args.width, args.height, &mut generator);

    let output = Output {
        all_states: args.states,
        active_states: args.activity,
        images: args.images,
    };

    protocol.execute(&mut lattice, &parameters, &mut generator, output);
}

fn main() {
    execute_protocol();
}
