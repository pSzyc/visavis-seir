#!/bin/bash

for width in {2..20}; do
    sed -i "s/HEIGHT: usize = .*/HEIGHT: usize = $width;/" src/lattice.rs;
    echo "compiling for length=$width"
    cargo build --release;
    echo "copying to ./target/bins/vis-a-vis-$width";
    cp ./target/release/vis-a-vis ./target/bins/vis-a-vis-$width;
done
