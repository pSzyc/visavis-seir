// QEIR, simulator of a monolayer of cells that hold a simple internal state and communicate when in contact
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use crate::cell::Cell;
use crate::compartment::Compartment::*;
use crate::lattice::Lattice;
use crate::rates::Rates;
use crate::subcompartments::Subcompartments;

#[derive(Debug, Copy, Clone)]
pub enum Event {
    CRate,
    EIncr,
    IIncr,
    RIncr,
}

impl Event {
    pub fn rate_coef(self, rates: &Rates) -> f64 {
        match self {
            Event::CRate => rates.c_rate,
            Event::EIncr => rates.e_incr,
            Event::IIncr => rates.i_incr,
            Event::RIncr => rates.r_incr,
        }
    }

    pub fn occur(
        event_i: usize,
        lattice: &mut Lattice,
        cell_i: usize,
        substates: &Subcompartments,
    ) -> Vec<usize> {
        let cell = &mut lattice.cells[cell_i];
        let compartments = &mut cell.compartments;
        let neighs = &lattice.neighborhoods[cell_i];
        macro_rules! increment {
            ($c:ident) => {
                compartments[$c as usize] += 1;
            };
        }
        macro_rules! set_zero {
            ($c:ident) => {
                compartments[$c as usize] = 0;
            };
        }
        macro_rules! current_cell {
            () => {
                vec![cell_i]
            };
        }
        macro_rules! current_cell_and_neighboring_cells {
            () => {
                vec![
                    cell_i, neighs[0], neighs[1], neighs[2], neighs[3], neighs[4], neighs[5],
                ]
            };
        }
        let event = Event::from_index(event_i);
        match event {
            Event::CRate => {
                debug_assert!(
                    Cell::is_zero(E, compartments)
                        && Cell::is_zero(I, compartments)
                        && Cell::is_zero(R, compartments)
                );
                increment!(E);
                current_cell!()
            }

            Event::EIncr => {
                if Cell::is_zero(E, compartments) {
                    increment!(E);
                    current_cell!()
                } else {
                    if substates.can_advance(E, compartments) {
                        increment!(E);
                        current_cell!()
                    } else {
                        debug_assert!(Cell::is_zero(I, compartments));
                        increment!(I);
                        set_zero!(E);
                        current_cell_and_neighboring_cells!()
                    }
                }
            }

            Event::IIncr => {
                if substates.can_advance(I, compartments) {
                    increment!(I);
                    current_cell!()
                } else {
                    debug_assert!(Cell::is_zero(R, compartments));
                    increment!(R);
                    set_zero!(I);
                    current_cell_and_neighboring_cells!()
                }
            }

            Event::RIncr => {
                if substates.can_advance(R, compartments) {
                    increment!(R);
                } else {
                    debug_assert!(Cell::is_zero(E, compartments) && Cell::is_zero(I, compartments));
                    set_zero!(R);
                }
                current_cell!()
            }
        }
    }

    #[inline]
    fn from_index(event_i: usize) -> Event {
        match event_i {
            0 => Event::CRate,
            1 => Event::EIncr,
            2 => Event::IIncr,
            3 => Event::RIncr,
            _ => {
                panic!("☠ @ event_i={}", event_i)
            }
        }
    }

    #[inline]
    pub fn to_index(self) -> usize {
        self as usize
    }
}
