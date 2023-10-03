#!/bin/bash
set -e

binary_dir=./target/bins/
update_existing=false

while getopts w:l:t:u flag
do
    case "${flag}" in
        w) width=${OPTARG};;
        l) length=${OPTARG};;
        f) binary_dir=${OPTARG};;
        u) update_existing=true;;
    esac
done

if [ -d $binary_dir ] && [ $update_existing = false ]; then
    echo $update_existing 
    echo "${binary_dir}/vis-a-vis-${width}-${length}: Directory exists, skipping"
else
    mkdir -p $binary_dir
    # sic! the names below don't match notation in the paper
    sed -i "s/HEIGHT: usize = .*/HEIGHT: usize = $width;/" src/lattice.rs;
    sed -i "s/WIDTH: usize = .*/WIDTH: usize = $length;/" src/lattice.rs;
    echo "compiling for length=$length"
    cargo build --release;
    sleep 1;
    cp ./target/release/vis-a-vis ${binary_dir}/vis-a-vis-${width}-${length};
fi

