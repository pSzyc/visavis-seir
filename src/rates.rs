// VIS-A-VIS, a simulator of Viral Infection Spread And Viral Infection Self-containment.
//
// Copyright (2022) Marek Kochanczyk & Frederic Grabowski (IPPT PAN, Warsaw).
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use std::fs;

use serde::{Deserialize, Serialize};
use serde_json::from_str;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Rates {
    pub c_rate: f64,
    pub e_incr: f64,
    pub i_incr: f64,
    pub r_incr: f64,
}

impl Rates {
    pub fn from_json_file(params_filename: &String) -> Self {
        let contents = fs::read_to_string(params_filename).expect("☠ 🕮 JSON");
        from_str(&contents).unwrap()
    }
}
