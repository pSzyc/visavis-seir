// QEIR, simulator of a monolayer of cells that hold a simple internal state and communicate when in contact
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use crate::cell::Cell;
use crate::compartment::{Compartment::{E, I, R}, NONQ_COMPARTMENTS_COUNT};
use crate::event::Event;
use crate::lattice::Lattice;
use crate::output::Output;
use crate::parameters::Parameters;
use crate::rates::Rates;
use crate::subcompartments::Subcompartments;
use crate::units::MIN;

use rand::{rngs::StdRng, Rng};
use std::fs::File;
use std::io::Write; // for .flush()
use threadpool::ThreadPool;

#[inline]
const fn ceil_pow2(i: u32) -> u32 {
    let mut v = i - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v + 1 + ((v == 0) as u32)
}

pub struct Simulation<'a> {
    propensities_tree_cell_index_base: usize,
    propensities_events_size: usize,
    propensities: Vec<Vec<f64>>,
    lattice: &'a mut Lattice,
}

impl<'a> Simulation<'a> {
    pub fn new(lattice: &'a mut Lattice) -> Self {
        Simulation {
            propensities_tree_cell_index_base: ceil_pow2(lattice.capacity as u32) as usize
                + lattice.capacity
                - 1
                - lattice.capacity,
            propensities_events_size: 1 + NONQ_COMPARTMENTS_COUNT,
            propensities: vec![],
            lattice,
        }
    }

    #[inline]
    fn unset_cell_event_prop(&mut self, cell_i: usize, event_i: usize) {
        let mut propens_i = self.propensities_tree_cell_index_base + cell_i;
        let rate = self.propensities[propens_i][event_i];
        debug_assert!(rate >= 0.);
        if rate > 0. {
            loop {
                self.propensities[propens_i][event_i] -= rate;
                if propens_i == 0 {
                    break;
                } else {
                    propens_i = (propens_i - 1) / 2
                }
            }
        }
    }

    #[inline]
    fn unset_cell_events_props(&mut self, cell_i: usize) {
        for event_i in 0..self.propensities_events_size {
            self.unset_cell_event_prop(cell_i, event_i)
        }
    }

    #[inline]
    fn set_event_propensity(&mut self, cell_i: usize, event_i: usize, rate: f64) {
        let mut propens_i = self.propensities_tree_cell_index_base + cell_i;
        loop {
            self.propensities[propens_i][event_i] += rate;
            if propens_i == 0 {
                break;
            } else {
                propens_i = (propens_i - 1) / 2
            }
        }
    }

    fn set_cell_events_props(&mut self, rates: &Rates, cell_i: usize) {
        let &cell = &self.lattice.cells[cell_i];
        if !cell.alive {
            if cfg!(debug_assertions) {
                for event_i in 0..self.propensities_events_size {
                    debug_assert!(
                        self.propensities[self.propensities_tree_cell_index_base + cell_i][event_i]
                            .abs() < 1.0e-6
                    );
                }
            }
            return;
        }

        let &cs = &cell.compartments;

        macro_rules! is_zero {
            ($m:ident) => {
                Cell::is_zero($m, &cs)
            };
        }

        macro_rules! set_ev_prop {
            ($rxn:ident) => {
                let r = Event::$rxn;
                let rate = r.rate_coef(rates);
                self.set_event_propensity(cell_i, r.to_index(), rate);
            };
        }

        let is_quiescent = is_zero!(E) && is_zero!(I) && is_zero!(R);
        if is_quiescent {
            let cell_i_neighborhood = self.lattice.neighborhoods[cell_i].clone();
            for neigh_cell_i in cell_i_neighborhood.iter() {
                let neigh_cs = self.lattice.cells[*neigh_cell_i].compartments;
                if !Cell::is_zero(I, &neigh_cs) {
                    set_ev_prop!(CRate);
                }
            }
        } else {
            if !is_zero!(E) {
                debug_assert!(is_zero!(I) && is_zero!(R));
                set_ev_prop!(EIncr);
            }
            if !is_zero!(I) {
                debug_assert!(is_zero!(E) && is_zero!(R));
                set_ev_prop!(IIncr);
            }
            if !is_zero!(R) {
                debug_assert!(is_zero!(E) && is_zero!(I));
                set_ev_prop!(RIncr);
            }
        }
    }

    fn compute_propensities(&mut self, rates: &Rates) {
        let propensities_tree_size =
            ceil_pow2(self.lattice.capacity as u32) as usize + self.lattice.capacity - 1;
        self.propensities = vec![vec![0.; self.propensities_events_size]; propensities_tree_size];
        for cell_i in 0..self.lattice.cells.len() {
            self.set_cell_events_props(rates, cell_i)
        }
    }

    fn find_event(&self, rho: f64) -> (usize, usize) {
        // select event class
        let mut acc = 0.;
        let mut event_i = 0;
        for ei in 0..self.propensities_events_size {
            acc += self.propensities[0][ei];
            if acc > rho {
                break;
            } else {
                event_i += 1
            }
        }

        // reuse random number
        let mut rho2 = rho - (acc - self.propensities[0][event_i]);
        debug_assert!(rho2 < self.propensities[0][event_i]);

        // select cell
        let mut cell_i = 0; // in-tree
        while cell_i < self.propensities_tree_cell_index_base {
            let next_left = 2 * cell_i + 1;
            let next_left_psum = self.propensities[next_left][event_i];
            if rho2 < next_left_psum {
                cell_i = next_left
            } else {
                rho2 -= next_left_psum;
                cell_i = next_left + 1
            }
        }

        debug_assert!(self.propensities[cell_i][event_i] > 0.);
        (cell_i - self.propensities_tree_cell_index_base, event_i)
    }

    pub fn run(
        &mut self,
        parameters: &Parameters,
        rng: &mut StdRng,
        tspan: (f64, f64),
        maybe_output: &Option<Output>,
        activity_horizontal_csv: &Option<File>,
        files_out_interval: f64,
        initial_frame_in_output_files: bool,
        output_workers: &ThreadPool,
    ) {
        let subcompartments = &Subcompartments {
            count: [
                parameters.e_subcompartments_count,
                parameters.i_subcompartments_count,
                parameters.r_subcompartments_count,
            ],
        };

        let rates = &Rates::new(
            parameters.c_rate,
            parameters.e_forward_rate,
            parameters.i_forward_rate,
            parameters.r_forward_rate,
        );

        self.compute_propensities(rates);
        let (mut t, mut t_next_files_out) = (
            tspan.0,
            tspan.0 + (if initial_frame_in_output_files {0.} else {files_out_interval}),
        );

        print!("{:.0}m:", t / MIN);
        std::io::stdout().flush().unwrap();

        loop {
            // check when next event occurs
            let sum_propens: f64 = self.propensities[0].iter().sum();
            if sum_propens > 0. {
                t += -(rng.gen_range(0.0..1.0) as f64).ln() / sum_propens; // exponential variate
            } else {
                t = tspan.1;
            }

            match maybe_output {
                Some(output) => {
                    while t >= t_next_files_out && t_next_files_out <= tspan.1 {
                        print!(".");
                        std::io::stdout().flush().unwrap();
                        if output.any_of_lattice() {
                            let lattice_snapshot = self.lattice.clone();
                            let output_clone = output.clone();
                            output_workers.execute(
                                move || lattice_snapshot.save_output_files(&output_clone, t_next_files_out)
                            );
                        }

                        match activity_horizontal_csv {
                            Some(csv_file) => {
                                debug_assert!(output.active_states);
                                self.lattice.save_activity_csv(t_next_files_out, csv_file);
                            }
                            None => {}
                        }

                        t_next_files_out += files_out_interval;
                    }
                }
                None => {}
            }

            // reached simulation end
            if t >= tspan.1 {
                print!(":{:.0}m ", tspan.1 / MIN);
                std::io::stdout().flush().unwrap();
                break;
            }

            // execute event if it happened in finite time
            if sum_propens > 0. {
                let (cell_i, event_i) = self.find_event(rng.gen_range(0.0..sum_propens));
                for cell_j in Event::occur(event_i, self.lattice, cell_i, subcompartments).iter() {
                    self.unset_cell_events_props(*cell_j);
                    self.set_cell_events_props(rates, *cell_j);
                }
            }
        } // loop
    } // run()
}
