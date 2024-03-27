// QEIR, simulator of a monolayer of cells that hold a simple internal state and communicate when in contact
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, Write};
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
    sequence::{delimited, pair, separated_pair, terminated, tuple},
};

use crate::commands::{initialize_front, run_simulation};
use crate::lattice::Lattice;
use crate::output::{Output, ACTIVITY_COLUMN_SUM_FILE_NAME};
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
        for line in reader.lines().map_while(Result::ok) {
            lines.push(line)
        }
        Protocol { commands: lines }
    }

    fn open_activity_file(output: &Output) -> Option<File> {
        if output.active_states {
            Some(
                OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(ACTIVITY_COLUMN_SUM_FILE_NAME)
                    .expect("☠ ☆ CSV[activity]"),
            )
        } else {
            None
        }
    }

    fn write_activity_file_header(activity_column_sum_file: &mut Option<File>, lattice: &Lattice) {
        match activity_column_sum_file {
            Some(ref mut csv_file) => {
                let mut header_vs: Vec<String> = vec!["time".to_string()];
                header_vs.extend(
                    (0..lattice.width)
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>(),
                );
                let header = header_vs.join(",") + "\n";
                csv_file
                    .write_all(header.as_bytes())
                    .expect("☠ ✏ CSV[activity]");
            }
            None => {}
        }
    }

    pub fn execute(
        &self,
        lattice: &mut Lattice,
        parameters: &Parameters,
        rng: &mut StdRng,
        output: Output,
    ) {
        let number = || pair::<_, _, _, (_, ErrorKind), _, _>(opt(char('-')), digit1);

        let in_unit_of = |u| move |s| -> Result<f64, ParseFloatError> { Ok(u * f64::from_str(s)?) };
        let seconds = || map_res(terminated(recognize(number()), char('s')), in_unit_of(SEC));
        let minutes = || map_res(terminated(recognize(number()), char('m')), in_unit_of(MIN));
        let hours = || map_res(terminated(recognize(number()), char('h')), in_unit_of(HOUR));
        let days = || map_res(terminated(recognize(number()), char('d')), in_unit_of(DAY));

        let time = || alt((seconds(), minutes(), hours(), days()));
        let timespan = || separated_pair(time(), tag("..."), time());
        let every = || delimited(char('['), time(), char(']'));
        let never = || tag("[]");
        let column = || map_res(recognize(number()), usize::from_str);

        let s = multispace1;
        let cmd_run = || tuple((tag("run"), s, timespan(), s, every()));
        let cmd_run_quiet = || tuple((tag("run"), s, timespan(), s, never()));
        let cmd_init_front = || tuple((tag("+front"), s, tag("at"), s, tag("column"), s, column()));

        let mut activity_file = Self::open_activity_file(&output);
        Self::write_activity_file_header(&mut activity_file, lattice);

        let n_threads = 1.max(num_cpus::get() - 1);  // leave one for the simulation
        let output_workers = threadpool::Builder::new().num_threads(n_threads).build();

        let mut initial_frame_in_output_files = true;

        for command in self.commands.iter() {
            if let Ok((_, (_, _, tspan, _, dt))) = cmd_run()(command) {
                run_simulation(
                    lattice,
                    parameters,
                    rng,
                    tspan,
                    &Some(output),
                    &activity_file,
                    dt,
                    initial_frame_in_output_files,
                    &output_workers,
                );
                initial_frame_in_output_files = false;
            } else if let Ok((_, (_, _, tspan, _, _))) = cmd_run_quiet()(command) {
                run_simulation(
                    lattice,
                    parameters,
                    rng,
                    tspan,
                    &None,
                    &activity_file,
                    -1.,
                    initial_frame_in_output_files,
                    &output_workers,
                );
                initial_frame_in_output_files = false;
            } else if let Ok((_, (_, _, _, _, _, _, column))) = cmd_init_front()(command) {
                initialize_front(lattice, column);
                initial_frame_in_output_files = true;
            } else {
                panic!("☠ @ command: {:?}", command);
            }
        }
        output_workers.join();
        println!();
    }
}
