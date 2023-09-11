// VIS-A-VIS, a simulator of Viral Infection Spread And Viral Infection Self-containment.
//
// Copyright (2022) Marek Kochanczyk & Frederic Grabowski (IPPT PAN, Warsaw).
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use crate::cell::Cell;
use crate::molecule::{Mol, N_MOLECULE_SPECIES};

use cairo::{Context, Format, ImageSurface};
use rand::{rngs::StdRng, seq::SliceRandom};
use std::f64::consts::PI;
use std::fs::{File, OpenOptions};
use std::io::{prelude::*, LineWriter};

type CellArray = [Cell; Lattice::CAPACITY];
type Neighborhoods = [[usize; Lattice::N_NEIGHBORS]; Lattice::CAPACITY];

#[derive(Clone)]
pub struct Lattice {
    pub neighborhoods: Neighborhoods,
    pub cells: CellArray,
}

impl Lattice {
    pub const N_NEIGHBORS: usize = 6; // fixed "kissing number" of the lattice, do not change

    pub const WIDTH: usize = 200; // if you increase this, you may also need to increase config::THREAD_STACK_SIZE!
    pub const HEIGHT: usize = 8;
    pub const CAPACITY: usize = Lattice::WIDTH * Lattice::HEIGHT;

    pub const OCCUPANCY: f64 = 1.0; // used as ceil(WIDTH * HEIGHT * the given fraction)

    // lattice output
    pub const NEIGHS_TO_FILE: bool = false; // whether lattice neighbor indices are to be dumped
    pub const IMAGE_RESOLUTION: u16 = 20; // default: 10
    pub const IMAGE_RECTANGULAR: bool = false; // if true, the parallelogram-shaped lattice is
                                              // right-to-left wrapped to form a rectangle

    pub fn new(rng: &mut StdRng) -> Self {
        Lattice {
            neighborhoods: Lattice::generate_neighborhods(),
            cells: Lattice::populate_cells(rng),
        }
    }

    fn generate_neighborhods() -> Neighborhoods {
        let mut nbhoods = [[usize::max_value(); Lattice::N_NEIGHBORS]; Lattice::CAPACITY];
        fn as_index(x: usize, y: usize) -> usize {
            x + y * Lattice::WIDTH
        }
        for (i, nbs) in nbhoods.iter_mut().enumerate() {
            let (x, y) = ((i % Lattice::WIDTH) as usize, (i / Lattice::WIDTH) as usize);
            let (east, west) = (
                (x + 1) % Lattice::WIDTH,
                (x + Lattice::WIDTH - 1) % Lattice::WIDTH,
            );
            let (south, north) = (
                (y + 1) % Lattice::HEIGHT,
                (y + Lattice::HEIGHT - 1) % Lattice::HEIGHT,
            );
            *nbs = [
                as_index(east, y),
                as_index(west, y),
                as_index(x, south),
                as_index(x, north),
                as_index(west, south),
                as_index(east, north),
            ];
        }
        if Lattice::NEIGHS_TO_FILE {
            let nbsf = File::create("neighbors.csv").expect("☠ ☆ neighs");
            let mut nbsf = LineWriter::new(nbsf);
            nbsf.write_all(b"left,right\n").expect("☠ ✏ neighs");
            for (i, nbs) in nbhoods.iter_mut().enumerate() {
                for nbi in nbs.iter() {
                    if nbi > &i {
                        nbsf.write_fmt(format_args!("{:},{:}\n", i, nbi))
                            .expect("☠ ✏ neighs");
                    }
                }
            }
        }
        nbhoods
    }

    fn populate_cells(rng: &mut StdRng) -> CellArray {
        let mut cells = [Cell {
            alive: true,
            molecules: [0; N_MOLECULE_SPECIES],
        }; Lattice::CAPACITY];
        let n_free_nodes = ((1.0 - Lattice::OCCUPANCY) * (cells.len() as f64)) as usize;
        (0..cells.len())
            .collect::<Vec<_>>()
            .choose_multiple(rng, n_free_nodes)
            .for_each(|i| cells[*i].alive = false);
        cells
    }

    fn save_png(&self, time: f64) {
        const IMG_SCALING: f64 = 20. * ((Lattice::IMAGE_RESOLUTION as f64) / 100.);
        const R: f64 = IMG_SCALING;
        const H: f64 = IMG_SCALING * 1.732_050 / 2.;
        const X0: f64 = 2. * H;
        const Y0: f64 = 1.5 * R;
        const HEIGHT: f64 = (1.5 * (Lattice::HEIGHT as f64) + 1.5) * R;
        const WIDTH: f64 = (2. * (Lattice::WIDTH as f64)
            + 1.
            + (if Lattice::IMAGE_RECTANGULAR { 2 } else { Lattice::HEIGHT }) as f64)
            * H;

        let sf = ImageSurface::create(Format::Rgb24, WIDTH as i32, HEIGHT as i32).unwrap();
        let cx = Context::new(&sf).unwrap();
        cx.set_source_rgb(0., 0., 0.);
        cx.paint().unwrap_or_else(|err| println!("☠ ✏ lattice: {:?}", err));
        cx.set_line_width(0.02 * IMG_SCALING);

        for cell_i in 0..Lattice::CAPACITY {
            // cell index --> its (x, y) coordinates
            let (mut i, j) = (cell_i % Lattice::WIDTH, cell_i / Lattice::WIDTH);
            if Lattice::IMAGE_RECTANGULAR {
                i = (i + j / 2) % Lattice::WIDTH
            }
            let (x, y) = (
                X0 + (2. * (i as f64) + (if Lattice::IMAGE_RECTANGULAR {j % 2} else {j} as f64)) * H,
                Y0 + 1.5 * (j as f64) * R,
            );

            // -- hexagon

            // contour
            cx.move_to(x, y + R * 0.99);
            for a in 2..=6 {
                let z = f64::from(a) * PI / 3.;
                cx.rel_line_to(R * 0.99 * z.sin(), R * 0.99 * z.cos())
            }
            cx.close_path();
            cx.set_source_rgb(0.1, 0.1, 0.1);
            cx.stroke_preserve().unwrap_or_else(|err| println!("☠ ✏ lattice: {:?}", err));

            // fill
            let (e, i, r) = (self.cells[cell_i].molecules[Mol::E as usize],
                             self.cells[cell_i].molecules[Mol::I as usize],
                             self.cells[cell_i].molecules[Mol::R as usize]);
            let is_susceptible = (e == 0) && (i == 0) && (r == 0);

            if is_susceptible {
                cx.set_source_rgb(0.85, 0.85, 0.85);
            } else {
                if e > 0 {
                    debug_assert!((i == 0) && (r == 0));
                    cx.set_source_rgb(0.2 + 0.1*(e as f64), 0.1 + 0.1*(e as f64), 0.);
                }
                if i > 0 {
                    debug_assert!((e == 0) && (r == 0));
                    cx.set_source_rgb(0.25 + 0.15*(i as f64), 0.0, 0.);
                }
                if r > 0 {
                    debug_assert!((e == 0) && (i == 0));
                    cx.set_source_rgb(0.35, 0.15 + 0.2*(r as f64), 0.85);
                }
            }
            cx.fill().unwrap_or_else(|err| println!("☠ ✏ lattice: {:?}", err));

        } // for each cell (lattice node)

        // write out image to a PNG file
        let png_fn = ["t_", &format!("{:0>6.0}", time), ".png"].concat();
        let mut png = File::create(png_fn).expect("☠ ☆ PNG.");
        sf.write_to_png(&mut png).expect("☠ ✏ PNG.");
    }

    fn save_csv(&self, time: f64) {
        // create and open CSV file for writing
        let csv_fn = ["t_", &format!("{:0>6.0}", time), ".csv"].concat();
        let mut csv = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(csv_fn)
            .expect("☠ ☆ CSV");

        // write out header
        let hdr = "id,E,I,R\n";
        csv.write_all(hdr.as_bytes()).expect("☠ ✏ CSV");

        // write out the state of each cell
        for cell_i in 0..Lattice::CAPACITY {
            let mut line: Vec<String> = vec![cell_i.to_string(),];
            macro_rules! count_s {
                ($m:ident) => {
                    self.cells[cell_i].molecules[$m as usize].to_string()
                };
            }
            for m in vec![Mol::E, Mol::I, Mol::R] {
                line.push(count_s!(m))
            }

            let mut line_s = line.join(",");
            line_s.push('\n');
            csv.write_all(line_s.as_bytes()).expect("☠ ✏ CSV");
        } // for each cell/lattice node
    }

    // save output file(s)
    pub fn out(&self, time: f64, dump_image: bool) {
        if dump_image {
            self.save_png(time);
        }
        self.save_csv(time);
    }
}

#[test]
fn test_lattice_neighborhood_reflectivity() {
    use rand::SeedableRng;
    let mut rng: StdRng = SeedableRng::from_seed([123; 32]);
    let nbhoods = &Lattice::new(&mut rng).neighborhoods;
    for i in 0..nbhoods.len() {
        assert_eq!(nbhoods[i].len(), Lattice::N_NEIGHBORS);
        assert_eq!(nbhoods[ nbhoods[i][0/*E */] ][1/*W */], i);
        assert_eq!(nbhoods[ nbhoods[i][2/*S */] ][3/*N */], i);
        assert_eq!(nbhoods[ nbhoods[i][4/*SW*/] ][5/*NE*/], i);
    }
}
