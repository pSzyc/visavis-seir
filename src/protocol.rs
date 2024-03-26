// QEIR, simulator of a monolayer of cells that hold a simple internal state and communicate when in contact
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use std::fs::File;
use std::io::{self, BufRead};
use std::num::ParseFloatError;
use std::path::Path;
use std::str::FromStr;

use rand::rngs::StdRng;

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1, multispace1},
    combinator::{map_res, opt, recognize},
    error::ErrorKind,
//  number::complete::double,
    sequence::{delimited, pair, separated_pair, terminated, tuple},
};

use crate::commands::{initialize_front, run_simulation, run_simulation_quietly};
use crate::lattice::Lattice;
use crate::output::Output;
use crate::parameters::Parameters;
use crate::units::{DAY, HOUR, MIN, SEC};

pub struct Protocol {
    pub commands: Vec<String>,
}

impl Protocol {
    pub fn from_text_file(protocol_file_path: &String) -> Self {
        let mut lines = Vec::<String>::new();
        let protocol_file_path = Path::new(protocol_file_path);
        let file = File::open(protocol_file_path).expect("☠ 🕮 Protocol");
        let reader = io::BufReader::new(file);
        for line in reader.lines().flatten() {
            lines.push(line)
        }
        Protocol { commands: lines }
    }

    pub fn execute(
        &self,
        lattice: &mut Lattice,
        parameters: &Parameters,
        rng: &mut StdRng,
        output: Output,
    ) {
     // let factor = || double::<&str, (_, ErrorKind)>;
        let number = || pair::<_, _, _, (_, ErrorKind), _, _>(opt(char('-')), digit1);

        let in_unit_of = |u| move |s| -> Result<f64, ParseFloatError> { Ok(u * f64::from_str(s)?) };
        let seconds = || map_res(terminated(recognize(number()), char('s')), in_unit_of(SEC));
        let minutes = || map_res(terminated(recognize(number()), char('m')), in_unit_of(MIN));
        let hours = || map_res(terminated(recognize(number()), char('h')), in_unit_of(HOUR));
        let days = || map_res(terminated(recognize(number()), char('d')), in_unit_of(DAY));

        let time = || alt((seconds(), minutes(), hours(), days()));
        let timespan = || separated_pair(time(), tag("..."), time());
        let every = || delimited(char('['), time(), char(']'));
        let square = || tag("[]");

        let cmd_run = || tuple((tag("run"), multispace1, timespan(), multispace1, every()));
        let cmd_run_quiet = || tuple((tag("run"), multispace1, timespan(), multispace1, square()));
        let cmd_init_front = || tuple((square(), char('!')));

        let mut out_init_frame = false; // whether initial frame in output

        for command in self.commands.iter() {
            if let Ok((_, (_, _, tspan, _, dt))) = cmd_run()(&command) {
                run_simulation(
                    lattice,
                    parameters,
                    rng,
                    tspan,
                    dt,
                    output,
                    out_init_frame,
                );
                out_init_frame = false;
            } else if let Ok((_, (_, _, tspan, _, _))) = cmd_run_quiet()(&command) {
                run_simulation_quietly(
                    lattice,
                    parameters,
                    rng,
                    tspan,
                    output,
                    out_init_frame,
                );
                out_init_frame = false;
            } else if let Ok((_, _)) = cmd_init_front()(&command) {
                initialize_front(lattice);
                out_init_frame = true;
            } else {
                panic!("☠ @ command: {:?}", command);
            }
        }
        println!();
    }
}
