// QEIR, simulator of a monolayer of cells that hold a simple internal state and communicate when in contact
//
// Copyright (2024) https://github.com/kochanczyk/qeir/CONTRIBUTORS.md.
// Licensed under the 3-Clause BSD license (https://opensource.org/licenses/BSD-3-Clause).

use crate::compartment::Compartment;
use crate::cell::{Cell, QUIESCENT_CELL};
use crate::output::{
    LATTICE_IMAGE_FILE_NAME_PREFIX,
    STATE_FILE_NAME_PREFIX,
    Output
};

use cairo::{Context, Format, ImageSurface, Matrix};
use rand::{rngs::StdRng, seq::SliceRandom};
use std::f64::consts::PI;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;

type CellArray = Vec<Cell>;
type Neighborhoods = Vec<Vec<usize>>;

#[derive(Clone)]
pub struct Lattice {
    pub cells: CellArray,
    pub neighborhoods: Neighborhoods,
    pub width: usize,
    pub height: usize,
    pub capacity: usize,
}

impl Lattice {
    pub const IMAGE_RESOLUTION: u16 = 100; // default: 100 (arbitrary units)
    pub const IMAGE_RECTANGULAR: bool = true; // if true then the parallelogram-shaped lattice
                                              // is right-to-left wrapped to form a rectangle
    pub fn new(width: usize, height: usize, rng: &mut StdRng) -> Self {
        Lattice {
            cells: Self::populate_cells(rng, width * height, 1.),
            neighborhoods: Self::generate_neighborhoods(width, height),
            width: width,
            height: height,
            capacity: width * height,
        }
    }

    fn populate_cells(rng: &mut StdRng, capacity: usize, occupancy: f64) -> CellArray {
        let mut cells = vec![
            Cell {
                alive: true,
                compartments: QUIESCENT_CELL,
            };
            capacity
        ];
        let n_free_nodes = ((1.0 - occupancy) * (cells.len() as f64)) as usize;
        (0..cells.len())
            .collect::<Vec<_>>()
            .choose_multiple(rng, n_free_nodes)
            .for_each(|i| cells[*i].alive = false);
        cells
    }

    fn generate_neighborhoods(width: usize, height: usize) -> Neighborhoods {
        let mut nbhoods = Neighborhoods::with_capacity(width * height);
        for i in 0..width * height {
            let (x, y) = ((i % height) as usize, (i / height) as usize);
            let (east, west) = ((x + 1) % height, (x + height - 1) % height);
            let (south, north) = ((y + 1) % width, (y + width - 1) % width);
            let as_index = |x: usize, y: usize| x + y * height;
            nbhoods.push(vec![
                as_index(east, y),
                as_index(west, y),
                as_index(x, south),
                as_index(x, north),
                as_index(west, south),
                as_index(east, north),
            ]);
        }
        nbhoods
    }

    fn save_png(&self, time: f64) {
        const IMG_SCALING: f64 = 20. * ((Lattice::IMAGE_RESOLUTION as f64) / 100.);
        const R: f64 = IMG_SCALING;
        const H: f64 = IMG_SCALING * 1.732_050 / 2.;
        const X0: f64 = 2. * H;
        const Y0: f64 = 1.5 * R;
        let png_width: f64 = (1.5 * (self.width as f64) + 1.5) * R;
        let png_height: f64 = (
            2. * (self.height as f64) + 1.
            + (if Lattice::IMAGE_RECTANGULAR {2} else {self.width}) as f64
        ) * H;

        let sf = ImageSurface::create(Format::Rgb24, png_width as i32, png_height as i32).expect("☠ ☆ lattice");
        let cx = Context::new(&sf).unwrap();
        let axes_swap = Matrix::new(0., 1., 1., 0., 0., 0.);
        cx.transform(axes_swap);

        cx.set_source_rgb(0., 0., 0.);
        cx.paint().unwrap_or_else(|err| println!("☠ ✏ lattice: {:?}", err));
        cx.set_line_width(0.02 * IMG_SCALING);

        for cell_i in 0..self.capacity {
            if !self.cells[cell_i].alive {
                continue;
            }

            // cell index --> cell (x, y) coordinates
            let (mut i, j) = (cell_i % self.height, cell_i / self.height);
            if Lattice::IMAGE_RECTANGULAR {
                i = (i + j / 2) % self.height
            }
            let (x, y) = (
                X0 + (2. * (i as f64)
                    + (if Lattice::IMAGE_RECTANGULAR { j % 2 } else { j } as f64))
                    * H,
                Y0 + 1.5 * (j as f64) * R,
            );

            // hexagon's contour
            cx.move_to(x, y + R * 0.99);
            for a in 2..=6 {
                let z = f64::from(a) * PI / 3.;
                cx.rel_line_to(R * 0.99 * z.sin(), R * 0.99 * z.cos())
            }
            cx.close_path();
            cx.set_source_rgb(0.1, 0.1, 0.1);
            cx.stroke_preserve().unwrap_or_else(|err| println!("☠ ✏ lattice: {:?}", err));

            // hexagon's fill
            let (e, i, r) = (
                self.cells[cell_i].compartments[Compartment::E as usize],
                self.cells[cell_i].compartments[Compartment::I as usize],
                self.cells[cell_i].compartments[Compartment::R as usize],
            );
            let is_quiescent = (e == 0) && (i == 0) && (r == 0);
            if is_quiescent {
                cx.set_source_rgb(0.85, 0.85, 0.85);
            } else {
                if e > 0 {
                    debug_assert!((i == 0) && (r == 0));
                    cx.set_source_rgb(0.2 + 0.1 * (e as f64), 0.1 + 0.1 * (e as f64), 0.);
                }
                if i > 0 {
                    debug_assert!((e == 0) && (r == 0));
                    cx.set_source_rgb(0.25 + 0.15 * (i as f64), 0.0, 0.);
                }
                if r > 0 {
                    debug_assert!((e == 0) && (i == 0));
                    cx.set_source_rgb(0.35, 0.15 + 0.2 * (r as f64), 0.85);
                }
            }
            cx.fill().unwrap_or_else(|err| println!("☠ ✏ lattice: {:?}", err));
        } // for each cell (lattice node)

        // write out an image to a PNG file
        let png_fn = [
            LATTICE_IMAGE_FILE_NAME_PREFIX, "t", &format!("{:0>6.0}", time), ".png"
        ].concat();
        let mut png = File::create(png_fn).expect("☠ ☆ PNG.");
        sf.write_to_png(&mut png).expect("☠ ✏ PNG.");
    }

    fn save_csv(&self, time: f64) {
        // create and open a CSV file for writing
        let csv_fn = [
            STATE_FILE_NAME_PREFIX, "t", &format!("{:0>6.0}", time), ".csv"
        ].concat();
        let mut csv = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(csv_fn)
            .expect("☠ ☆ CSV");

        // write out the header
        let hdr = "id,w,h,E,I,R\n";
        csv.write_all(hdr.as_bytes()).expect("☠ ✏ CSV");

        // write out the state of each cell
        for cell_i in 0..self.capacity {
            let cell_h = cell_i / self.height;
            let cell_w = cell_i % self.height;

            let mut line: Vec<String> = vec![cell_i.to_string(), cell_w.to_string(), cell_h.to_string()];
            macro_rules! count_s {
                ($c:ident) => {
                    self.cells[cell_i].compartments[$c as usize].to_string()
                };
            }
            for c in vec![Compartment::E, Compartment::I, Compartment::R] {
                line.push(count_s!(c))
            }

            let mut line_s = line.join(",");
            line_s.push('\n');
            csv.write_all(line_s.as_bytes()).expect("☠ ✏ CSV");
        } // for each cell/lattice node
    }

    pub fn save_activity_csv(&self, time: f64, mut csv: &File) {
        macro_rules! in_compartment {
            ($c:expr, $ci:ident) => {
                self.cells[$ci].compartments[$c as usize] > 0
            };
        }

        let is_active = |x, y| -> bool {
            let cell_index: usize = x*self.height + y;
            in_compartment!(Compartment::E, cell_index) || in_compartment!(Compartment::I, cell_index)
        };

        let mut line_vs: Vec<String> = vec![time.to_string()];
        for x in 0..self.width {
            let mut activity_vert_sum: i32 = 0;
            for y in 0..self.height {
                activity_vert_sum += is_active(x, y) as i32
            }
            line_vs.push(activity_vert_sum.to_string())
        }
        let line = line_vs.join(",") + "\n";

        csv.write_all(line.as_bytes()).expect("☠ ✏ CSV");
    }

    pub fn save_output_files(&self, output: &Output, time: f64) {
        if output.all_states {
            self.save_csv(time);
        }
        if output.images {
            self.save_png(time);
        }
    }
}

#[test]
fn test_lattice_neighborhood_reflectivity() {
    const KISSING_NUMBER: usize = 6;
    use rand::SeedableRng;
    let mut rng: StdRng = SeedableRng::from_seed([123; 32]);
    let nbhoods = &Lattice::new(24, 24, &mut rng).neighborhoods;
    for i in 0..nbhoods.len() {
        assert_eq!(nbhoods[i].len(), KISSING_NUMBER);
        assert_eq!(nbhoods[ nbhoods[i][0/*E */] ][1/*W */], i);
        assert_eq!(nbhoods[ nbhoods[i][2/*S */] ][3/*N */], i);
        assert_eq!(nbhoods[ nbhoods[i][4/*SW*/] ][5/*NE*/], i);
    }
}
