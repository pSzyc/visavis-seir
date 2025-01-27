# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk

    
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from shutil import rmtree

import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) # in order to be able to import from scripts.py

from scripts.simulation import run_simulation
from scripts.utils import starmap, random_name
from scripts.defaults import TEMP_DIR, PARAMETERS_DEFAULT
from scripts.binary import activity_to_arrival_times, arrival_times_to_dataset, get_entropy


def perform_single(
    channel_width, 
    channel_length, 
    interval, 
    n_pulses, 
    simulation_id,  
    parameters=PARAMETERS_DEFAULT, 
    logging_interval=5, 
    interval_after=None, 
    outdir=None,  
    save_iterations=False, 
    min_distance_between_peaks=6, 
    ends=[0],
    ):
    
    sim_dir = Path(f"{TEMP_DIR}/qeir/periodic/" + random_name(12))

    if interval_after is None:
        interval_after = int(2 * 3.6*channel_length+200)

    result = run_simulation(
        parameters=parameters,
        channel_width=channel_width,
        channel_length=channel_length,
        pulse_intervals=n_pulses * [interval] + [interval_after],
        logging_interval=logging_interval,
        save_activity=save_iterations,
        outdir=outdir,

        seed=simulation_id + 19,
        sim_root=sim_dir,
        )

    rmtree(sim_dir)


    # if len(ends) == 1:
    #     arrival_times = pd.Series(activity_to_arrival_times(result.activity, min_distance_between_peaks=min_distance_between_peaks, end=ends[0]-1)).to_frame()
    # else: 
    arrival_times = pd.concat([
        pd.Series(activity_to_arrival_times(result.activity, min_distance_between_peaks=min_distance_between_peaks, end=end-1)).to_frame()
        for end in ends
        ], 
        names=['end'], keys=ends)
        
    return arrival_times
    


def generate_dataset(
    interval,
    channel_width,
    channel_length,
    n_simulations,
    save_iterations=False,
    outdir=None,
    ends=[0],
    use_cached=False,
    processes=None,
    **kwargs,
):

    if outdir: 
        outpath = outdir / ('arrival_times.csv')


    if outdir and use_cached and outpath.exists():
        arrival_times = pd.read_csv(outpath).set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'end', 'pulse_id'])
        if use_cached == 'always' or sorted(tuple(idx) for _, idx in arrival_times.reset_index()[['channel_width', 'channel_length', 'interval', 'simulation_id', 'end']].drop_duplicates().iterrows()) == sorted(product([channel_width], [channel_length], [interval], range(n_simulations), ends)):
            return arrival_times


    arrival_times = pd.concat(starmap(
        perform_single,
        [
            dict(
                interval=interval,
                channel_width=channel_width,
                channel_length=channel_length,
                simulation_id=simulation_id,
                outdir=outdir,
                save_iterations=save_iterations,
                ends=ends,
                **kwargs,
                )
            for simulation_id in range(n_simulations)
        ], processes=processes,
    ), names=['simulation_id'], keys=list(range(n_simulations)))

    arrival_times.index.names = ['simulation_id', 'end', 'pulse_id']

    arrival_times['channel_width'] = channel_width
    arrival_times['channel_length'] = channel_length
    arrival_times['interval'] = interval

    arrival_times = arrival_times.reset_index().set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'end', 'pulse_id'])

    if outdir:
        outdir.mkdir(exist_ok=True, parents=True)
        arrival_times.to_csv(outpath)

    return arrival_times


def generate_dataset_batch(
        channel_lengths,
        channel_widths,
        intervals, 
        outdir=None,
        processes=None,
        **kwargs
    ):

    # outpath = outdir / 'dataset.csv' if outdir else None

    dataset_parts = []

    for channel_length, channel_width, interval in product(channel_lengths, channel_widths, intervals):
        print(f"l={channel_length} w={channel_width} i={interval}", end='  ', flush=True)
        data = generate_dataset(
            interval=interval,
            channel_width=channel_width,
            channel_length=channel_length,
            processes=processes,
            outdir=outdir and outdir / f"l-{channel_length}-w-{channel_width}-i-{interval}",
            **kwargs
        )
        dataset_parts.append(data.reset_index())
    dataset = pd.concat(dataset_parts, ignore_index=True)
    # if outpath:
    #     dataset.to_csv(outpath)
    return dataset

