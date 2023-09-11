// VIS-A-VIS, a simulator of Viral Infection Spread And Viral Infection Self-containment.
//
// Copyright (2022) Marek Kochanczyk & Frederic Grabowski (IPPT PAN, Warsaw).
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use crate::cell::Cell;
use crate::event::Event;
use crate::lattice::Lattice;
use crate::molecule::Mol::{E, I, R};
use crate::molecule::N_MOLECULE_SPECIES;
use crate::rates::Rates;
use crate::units::MIN;

use rand::{rngs::StdRng, Rng};
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

const PROPENS_TREE_SIZE: usize =
    ceil_pow2(Lattice::CAPACITY as u32) as usize + Lattice::CAPACITY - 1;
const PROPENS_TREE_CELL_INDEX_BASE: usize = PROPENS_TREE_SIZE - Lattice::CAPACITY;
const PROPENS_EVENTS_SIZE: usize = 2 * N_MOLECULE_SPECIES + 1; // +molecule,-molecule, and 1 for Die
type Propensities = [[f64; PROPENS_EVENTS_SIZE]; PROPENS_TREE_SIZE];

pub struct Simulation {}

impl Simulation {
    #[inline]
    fn unset_cell_event_prop(propens: &mut Propensities, cell_i: usize, event_i: usize) {
        let mut propens_i = PROPENS_TREE_CELL_INDEX_BASE + cell_i;
        let rate = propens[propens_i][event_i];
        debug_assert!(rate >= 0.);
        if rate > 0. {
            loop {
                propens[propens_i][event_i] -= rate;
                if propens_i == 0 { break; } else { propens_i = (propens_i - 1) / 2 }
            }
        }
    }

    #[inline]
    fn unset_cell_events_props(propens: &mut Propensities, cell_i: usize) {
        for event_i in 0..PROPENS_EVENTS_SIZE {
            Simulation::unset_cell_event_prop(propens, cell_i, event_i)
        }
    }

    #[inline]
    fn set_event_propensity(propens: &mut Propensities, cell_i: usize, event_i: usize, rate: f64) {
        let mut propens_i = PROPENS_TREE_CELL_INDEX_BASE + cell_i;
        loop {
            propens[propens_i][event_i] += rate;
            if propens_i == 0 { break; } else { propens_i = (propens_i - 1) / 2 }
        }
    }

    fn set_cell_events_props(
        propens: &mut Propensities,
        lattice: &Lattice,
        rates: &Rates,
        cell_i: usize,
    ) {
        let &cell = &lattice.cells[cell_i];
        if !cell.alive {
            if cfg!(debug_assertions) {
                for event_i in 0..PROPENS_EVENTS_SIZE {
                    debug_assert!(
                        propens[PROPENS_TREE_CELL_INDEX_BASE + cell_i][event_i].abs() < 1.0e-6
                    );
                }
            }
            return;
        }

        let &ms = &cell.molecules;

        macro_rules! is_zero      { ($m:ident) => { Cell::is_zero($m, &ms) }; }

        macro_rules! set_ev_prop {
            ($rxn:ident, $rate_mul:expr, $rate_add:expr) => {
                let r = Event::$rxn;
                let rate = r.rate_coef(rates) * $rate_mul + $rate_add;
                Simulation::set_event_propensity(propens, cell_i, r.to_index(), rate);
            };
            ($rxn:ident, $rate_mul:expr) => {
                set_ev_prop!($rxn, $rate_mul, 0.)
            };
            ($rxn:ident) => {
                set_ev_prop!($rxn, 1., 0.)
            };
        }

        let is_susceptible = is_zero!(E) && is_zero!(I) && is_zero!(R);
        if is_susceptible {
            for neigh_cell_i in lattice.neighborhoods[cell_i].iter() {
                let neigh_ms = lattice.cells[*neigh_cell_i].molecules;
                let has_infectious_neigh = Cell::can_decrease(I, &neigh_ms);
                if has_infectious_neigh { set_ev_prop!(CRate); }
            }
        } else {
            if ! is_zero!(E) { set_ev_prop!(EIncr); }
            if ! is_zero!(I) { set_ev_prop!(IIncr); }
            if ! is_zero!(R) { set_ev_prop!(RIncr); }
        }
    }

    fn compute_propensities(
        lattice: &Lattice,
        rates: &Rates,
    ) -> Propensities {
        let mut propens: Propensities = [[0.; PROPENS_EVENTS_SIZE]; PROPENS_TREE_SIZE];
        for cell_i in 0..lattice.cells.len() {
            Simulation::set_cell_events_props(&mut propens, lattice, rates, cell_i)
        }
        propens
    }

    fn find_event(propens: &Propensities, rho: f64) -> (usize, usize) {
        // select event class
        let mut acc = 0.;
        let mut event_i = 0;
        for ei in 0..PROPENS_EVENTS_SIZE {
            acc += propens[0][ei];
            if acc > rho {
                break;
            } else {
                event_i += 1
            }
        }

        // reuse random number
        let mut rho2 = rho - (acc - propens[0][event_i]);
        debug_assert!(rho2 < propens[0][event_i]);

        // select cell
        let mut cell_i = 0; // in-tree
        while cell_i < PROPENS_TREE_CELL_INDEX_BASE {
            let next_left = 2 * cell_i + 1;
            let next_left_psum = propens[next_left][event_i];
            if rho2 < next_left_psum {
                cell_i = next_left
            } else {
                rho2 -= next_left_psum;
                cell_i = next_left + 1
            }
        }

        debug_assert!(propens[cell_i][event_i] > 0.);
        (cell_i - PROPENS_TREE_CELL_INDEX_BASE, event_i)
    }


    pub fn simulate(
        lattice: &mut Lattice,
        rates: &Rates,
        rng: &mut StdRng,
        tspan: (f64, f64),
        files_out: bool,
        images_out: bool,
        files_out_interval: f64,
        ifni_secretion: bool,
        in_sep_thread: bool,
        init_frame_out: bool,
        workers: &Option<ThreadPool>,
    ) {
        // (currently, these 3 parameters are redundant)
        debug_assert!(in_sep_thread == workers.is_none());
        debug_assert!(in_sep_thread == !ifni_secretion);

        let mut propens = Simulation::compute_propensities(lattice, rates);
        let (mut t, mut t_next_files_out) = (
            tspan.0,
            tspan.0 + (if init_frame_out { 0. } else { files_out_interval }),
        );
        if !in_sep_thread {
            print!("{:.0}m:", t / MIN);
            std::io::stdout().flush().unwrap()
        }
        loop {
            // if t >= t_next_print_out && !in_sep_thread { t_next_print_out += 1.*HOUR }
            if files_out && t >= t_next_files_out {
                if !in_sep_thread {
                    // spawn in a separate thread
                    print!(".");
                    std::io::stdout().flush().unwrap();
                    let la = lattice.clone();
                    workers.as_ref().unwrap().execute(move || { la.out(t, images_out)});
                }
                t_next_files_out += files_out_interval;
            }
            if t >= tspan.1 {
                print!(":{:.0}m ", tspan.1 / MIN);
                std::io::stdout().flush().unwrap();
                break;
            }
            let sum_propens: f64 = propens[0].iter().sum();
            t += -(rng.gen_range(0.0..1.0) as f64).ln() / sum_propens; // exponential variate
            if sum_propens > 0. {
                let (cell_i, event_i) =
                    Simulation::find_event(&propens, rng.gen_range(0.0..sum_propens));
                for cell_j in Event::occur(event_i, lattice, cell_i).iter() {
                    Simulation::unset_cell_events_props(&mut propens, *cell_j);
                    Simulation::set_cell_events_props(
                        &mut propens,
                        lattice,
                        rates,
                        *cell_j,
                    );
                }
            }
        } // loop
    } // simulate()
}
