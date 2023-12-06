// VIS-A-VIS, a simulator of Viral Infection Spread And Viral Infection Self-containment.
//
// Copyright (2022) Marek Kochanczyk & Frederic Grabowski (IPPT PAN, Warsaw).
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use std::fs;

use serde::{Deserialize, Serialize};
use serde_json::from_str;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct States {
    pub n_e: u8,
    pub n_i: u8,
    pub n_r: u8,
}

impl States {
    pub fn from_json_file(states_filename: &String) -> Self {
        let contents = fs::read_to_string(states_filename).expect("☠ 🕮 JSON");
        from_str(&contents).unwrap()
    }
}
