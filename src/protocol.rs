// VIS-A-VIS, a simulator of Viral Infection Spread And Viral Infection Self-containment.
//
// Copyright (2022) Marek Kochanczyk & Frederic Grabowski (IPPT PAN, Warsaw).
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use std::fs::{OpenOptions, File};
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
    number::complete::double,
    sequence::{delimited, pair, separated_pair, terminated, tuple},
};

use crate::commands::{initialize_epidemics, run_simulation, run_simulation_quietly};
use crate::lattice::Lattice;
use crate::rates::Rates;
use crate::units::{SEC, MIN, HOUR, DAY};

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
        rates: &Rates,
        rng: &mut StdRng,
        out_images: bool,
        out_activity: bool,
        out_states: bool,
    ) {
        let factor = || double::<&str, (_, ErrorKind)>;
        let number = || pair::<_, _, _, (_, ErrorKind), _, _>(opt(char('-')), digit1);

        let in_unit_of = |u| move |s| -> Result<f64, ParseFloatError> { Ok(u * f64::from_str(s)?) };
        let seconds = || map_res( terminated(recognize(number()), char('s')), in_unit_of(SEC));
        let minutes = || map_res( terminated(recognize(number()), char('m')), in_unit_of(MIN));
        let hours = || map_res( terminated(recognize(number()), char('h')), in_unit_of(HOUR));
        let days = || map_res( terminated(recognize(number()), char('d')), in_unit_of(DAY));

        let time = || alt((seconds(), minutes(), hours(), days()));
        let timespan = || separated_pair(time(), tag("..."), time());
        let every = || delimited(char('['), time(), char(']'));
        let never = || tag("[]");

        let cmd_run = || tuple((tag("run"), multispace1, timespan(), multispace1, every()));
        let cmd_run_quiet = || tuple((tag("run"), multispace1, timespan(), multispace1, never()));
        let cmd_init_epidemics = || tuple((tag("+batsoup"), multispace1, factor()));

        let mut activity_csv: Option<&File> = None;
        let mut csv: File;
        if out_activity {
            // create and open CSV file for writing
            csv = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open("activity.csv")
                .expect("☠ ☆ CSV");
            // write out header
            let mut hdr_s: Vec<String> = vec!["time".to_string()];
            for h in 0..Lattice::HEIGHT {
                hdr_s.push(h.to_string())
            }
            let hdr = hdr_s.join(",") + "\n";
            csv.write_all(hdr.as_bytes()).expect("☠ ✏ CSV");
            activity_csv = Some(&csv);
        }

        let mut out_init_frame = false; // whether initial frame in output
        for command in self.commands.iter() {
            if let Ok((_, (_, _, tspan, _, dt))) = cmd_run()(&command) {
                run_simulation(lattice, rates, rng, tspan, out_images, out_states, dt, out_init_frame, activity_csv);
                out_init_frame = false;
            } else if let Ok((_, (_, _, tspan, _, _))) = cmd_run_quiet()(&command) {
                run_simulation_quietly(lattice, rates, rng, tspan, out_images, out_states, out_init_frame, activity_csv);
                out_init_frame = false;
            } else if let Ok((_, _)) = cmd_init_epidemics()(&command) {
                initialize_epidemics(lattice);
                out_init_frame = true;
            } else {
                panic!("☠ @ command: {:?}", command);
            }
        }
        println!();
    }
}
