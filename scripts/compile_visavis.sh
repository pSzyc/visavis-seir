#!/bin/bash
set -e

usage="$(basename "$0") [-h] -w width -l length [-t target_directory] [-u] 
Compile visavis and copy to target_directory
Options:
    -w  channel width
    -l  channel length
    -t  target directory
    -u  (flag) update existing build
    -h  show this help"

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

if [ -d $binary_dir/vis-a-vis-${width}-${length} ] && [ $update_existing = false ]; then
    echo "${binary_dir}/vis-a-vis-${width}-${length}: Directory exists, skipping"
else
    mkdir -p $binary_dir
    sed -i "s/HEIGHT: usize = .*/HEIGHT: usize = $length;/" src/lattice.rs;
    sed -i "s/WIDTH: usize = .*/WIDTH: usize = $width;/" src/lattice.rs;
    echo "compiling for length=$length"
    cargo build --release;
    sleep 1;
    echo "copying to $binary_dir/vis-a-vis-$width-$length";
    cp ./target/release/vis-a-vis ${binary_dir}/vis-a-vis-${width}-${length};
fi

