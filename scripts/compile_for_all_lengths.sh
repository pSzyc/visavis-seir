#!/bin/bash
set -e

BINARY_DIR=./target/bins/

mkdir -p $BINARY_DIR

width=6
for length in {160,400,600,840}; do 
    sed -i "s/HEIGHT: usize = .*/HEIGHT: usize = $length;/" src/lattice.rs;
    sed -i "s/WIDTH: usize = .*/WIDTH: usize = $width;/" src/lattice.rs;
    echo "compiling for length=$length"
    cargo build --release;
    sleep 1;
    echo "copying to $binary_dir/vis-a-vis-$width-$length";
    cp ./target/release/vis-a-vis ${BINARY_DIR}/vis-a-vis-${width}-${length};
done
