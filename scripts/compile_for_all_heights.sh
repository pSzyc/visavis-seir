#!/bin/bash

for h in {2..20}; do
    sed -i "s/HEIGHT: usize = .*/HEIGHT: usize = $h;/" src/lattice.rs;
    echo "compiling for height=$h"
    cargo build --release;
    cp ./target/release/vis-a-vis ./target/bins/vis-a-vis-$h;
done
