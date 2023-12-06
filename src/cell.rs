// VIS-A-VIS, a simulator of Viral Infection Spread And Viral Infection Self-containment.
//
// Copyright (2022) Marek Kochanczyk & Frederic Grabowski (IPPT PAN, Warsaw).
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use crate::molecule::Mol;
use crate::molecule::N_MOLECULE_SPECIES;

pub type MolArray = [u8; N_MOLECULE_SPECIES];

#[derive(Clone, Copy)]
pub struct Cell {
    pub alive: bool,
    pub molecules: MolArray,
}

impl Cell {
    #[inline]
    pub fn is_zero(m: Mol, ms: &MolArray) -> bool {
        let i = m as usize;
        ms[i] == 0
    }
}
