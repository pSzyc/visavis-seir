// QEIR, simulator of a monolayer of directly communicating cells which hold a simple internal state
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

#[derive(Debug, Clone)]
pub struct Rates {
    pub c_rate: f64,
    pub e_incr: f64,
    pub i_incr: f64,
    pub r_incr: f64,
}

impl Rates {
    pub fn new(c_rate: f64, e_incr: f64, i_incr: f64, r_incr: f64) -> Self {
        Rates {
            c_rate: c_rate,
            e_incr: e_incr,
            i_incr: i_incr,
            r_incr: r_incr,
        }
    }
}
