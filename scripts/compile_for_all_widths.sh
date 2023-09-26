#!/bin/bash
set -e

BINARY_DIR=./target/bins/

mkdir -p $BINARY_DIR

width=6
for length in {30,45,70,100,130,200,250,300,500,700,1000}; do 
    # sic! the names below don't match notation in the paper
    sed -i "s/HEIGHT: usize = .*/HEIGHT: usize = $width;/" src/lattice.rs;
    sed -i "s/WIDTH: usize = .*/WIDTH: usize = $length;/" src/lattice.rs;
    echo "compiling for length=$length"
    cargo build --release;
    sleep 1;
    cp ./target/release/vis-a-vis ${BINARY_DIR}/vis-a-vis-${width}-${length};
done
