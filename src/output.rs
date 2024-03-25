// QEIR, simulator of a monolayer of directly communicating cells which hold a simple internal state
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

pub const STATE_FILE_NAME_PREFIX: &'static str = "state_";
pub const LATTICE_IMAGE_FILE_NAME_PREFIX: &'static str = "lattice_";
pub const ACTIVITY_HORIZONTAL_FILE_NAME: &'static str = "activity_horizontal.csv";


#[derive(Clone, Copy)]
pub struct Output {
    pub all_states: bool,    // one text file per output time point
    pub active_states: bool, // one text file per whole simulation
    pub images: bool,        // one image file per output time point
}

impl Output {
    #[inline]
    pub fn any(&self) -> bool {
        self.all_states || self.active_states || self.images
    }
}