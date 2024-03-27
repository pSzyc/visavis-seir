// QEIR, simulator of a monolayer of cells that hold a simple internal state and communicate when in contact
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

pub const STATE_FILE_NAME_PREFIX: &'static str = "state_";
pub const LATTICE_IMAGE_FILE_NAME_PREFIX: &'static str = "lattice_";
pub const ACTIVITY_COLUMN_SUM_FILE_NAME: &'static str = "activity_column_sum.csv";


#[derive(Clone, Copy)]
pub struct Output {
    pub all_states: bool,    // one text file per output time point
    pub active_states: bool, // one text file per whole simulation
    pub images: bool,        // one image file per output time point
}

impl Output {
    #[inline]
    pub fn any_of_lattice(&self) -> bool { self.all_states || self.images }
}
