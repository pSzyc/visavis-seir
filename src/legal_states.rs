// VIS-A-VIS, a simulator of Viral Infection Spread And Viral Infection Self-containment.
//
// Copyright (2022) Marek Kochanczyk & Frederic Grabowski (IPPT PAN, Warsaw).
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use crate::molecule::Mol;
use crate::cell::MolArray;

#[derive(Copy, Clone)]
pub struct LegalStates {
    pub max: MolArray,
    pub min: MolArray,
}

impl LegalStates {

    #[inline]
    pub fn can_increase(self, m: Mol, ms: &MolArray) -> bool {
        let i = m as usize;
        ms[i] < self.max[i]
    }

    #[inline]
    pub fn can_decrease(self, m: Mol, ms: &MolArray) -> bool {
        let i = m as usize;
        ms[i] > self.min[i]
    }
}
