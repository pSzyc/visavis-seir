#!/bin/bash

for width in {400..800..100}; do
    sed -i "s/WIDTH: usize = .*/WIDTH: usize = $width;/" src/lattice.rs;
    echo "compiling for width=$width"
    cargo build --release;
    cp ./target/release/vis-a-vis ./target/bins/vis-a-vis-w-$width;
done
