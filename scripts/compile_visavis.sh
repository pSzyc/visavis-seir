#!/bin/bash
set -e

usage="$(basename "$0") [-h] [-t target_directory] [-u] 
Compile visavis and copy to target_directory
Options:
    -t  target directory
    -u  (flag) update existing build
    -h  show this help"

binary_dir=./target/release/
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

if [ -d $binary_dir/vis-a-vis ] && [ $update_existing = false ]; then
    echo "${binary_dir}/vis-a-vis: Build exists, skipping"
else
    mkdir -p $binary_dir
    echo "compiling"
    cargo build --release;
    sleep 1;
    echo "copying to $binary_dir/vis-a-vis";
fi

