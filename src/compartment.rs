// QEIR, simulator of a monolayer of cells that hold a simple internal state and communicate when in contact
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

pub enum Compartment {
    // The quiescent compartment (Q) is implicit.
    E, // ~ "exposed"
    I, // ~ "infectious"
    R, // ~ "resistant"
}

pub const NONQ_COMPARTMENTS_COUNT: usize = 3;
