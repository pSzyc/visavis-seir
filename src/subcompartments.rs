// QEIR, simulator of a monolayer of cells that hold a simple internal state and communicate when in contact
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use crate::compartment::Compartment;
use crate::cell::Compartments;

#[derive(Copy, Clone)]
pub struct Subcompartments {
    pub count: Compartments,
}

impl Subcompartments {
    #[inline]
    pub fn can_advance(self, c: Compartment, cs: &Compartments) -> bool {
        let i = c as usize;
        cs[i] < self.count[i]
    }
}
