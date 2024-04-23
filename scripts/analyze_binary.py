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
from scripts.binary import activity_to_arrival_times, arrival_times_to_dataset, get_entropy, plot_scan
from scripts.analyze_velocity import get_velocity
from scripts.optimizer import get_optimum_from_scan


def make_binary_protocol(interval, n_slots, p=0.5, seed=0):
    random_generator = np.random.default_rng(seed)
    pulse_bits = random_generator.choice(2, size=n_slots, p=[1.-p, p])
    pulse_bits[0] = 1

    locs_1, = np.nonzero(pulse_bits)

    pulse_times = interval * locs_1
    pulse_intervals = pulse_times[1:] - pulse_times[:-1]
    
    return pulse_times, pulse_intervals


def perform_single(
        interval, 
        channel_width, 
        channel_length,
        n_slots, 
        duration, 
        offset, 
        n_margin, 
        n_nearest, 
        min_distance_between_peaks,
        sim_dir, 
        simulation_id, 
        parameters=PARAMETERS_DEFAULT, 
        p=0.5,
    ):
    departure_times, pulse_intervals = make_binary_protocol(interval, n_slots, p=p, seed=19+simulation_id)

    result = run_simulation(
            parameters=parameters,
            width=channel_width,
            length=channel_length,
            duration=duration,
            pulse_intervals=list(pulse_intervals) + [1500],
            verbose=False,
            sim_root=sim_dir,
            sim_dir_name=f"sim-{simulation_id}",
            seed=simulation_id+19,
            states=False,
            activity=True,
    )


    # print('Result_loaded')
    arrival_times = activity_to_arrival_times(result.activity, min_distance_between_peaks=min_distance_between_peaks)
    # print('Arrivals_computed')
        
    dataset_part = arrival_times_to_dataset(
            interval=interval,
            departure_times=departure_times,
            arrival_times=arrival_times,
            n_nearest=n_nearest, 
            n_margin=n_margin,
            offset=offset,
        )
    # print('Data set constructed')

    dataset_part['simulation_id'] = simulation_id

    return dataset_part


def generate_dataset(
    interval,
    n_slots,
    n_simulations,
    channel_width,
    channel_length,
    parameters=PARAMETERS_DEFAULT,
    duration=1,
    v=None,
    offset=None,
    p=0.5,
    outdir=None,
    velocity_cache_dir=Path(__file__).parent.parent / 'data' / 'velocity',
    n_margin=4,
    n_nearest=4,
    min_distance_between_peaks=20,
    processes=None,
):

    # TEMP_DIR = "/tmp"
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    sim_dir = Path(f"{TEMP_DIR}/qeir/binary/" + random_name(12))


    if v is None:
        v = get_velocity(channel_width, channel_length, parameters, velocity_cache_dir=velocity_cache_dir)

    if offset is None:
        offset = channel_length / v

   
    dataset = pd.concat(starmap(
        perform_single,
        [
            dict(
                interval=interval,
                channel_width=channel_width,
                channel_length=channel_length,
                n_slots=n_slots,
                duration=duration,
                offset=offset,
                parameters=parameters,
                p=p,
                n_margin=n_margin,
                n_nearest=n_nearest,
                sim_dir=sim_dir,
                simulation_id=simulation_id,
                min_distance_between_peaks=min_distance_between_peaks,
                )
            for simulation_id in range(n_simulations)
        ], processes=processes,
    ))


    dataset['channel_width'] = channel_width
    dataset['channel_length'] = channel_length
    dataset['interval'] = interval

    rmtree(sim_dir)
    dataset.index.name = 'pulse_id'
    if outdir:
        dataset.reset_index().set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'pulse_id']).to_csv(outdir / 'dataset.csv')

    return dataset


def generate_dataset_batch(
        channel_lengths,
        channel_widths,
        intervals, 
        outdir=None,
        v=None,
        use_cached=False,
        processes=None,
        **kwargs
    ):

    # outpath = outdir / 'dataset.csv' if outdir else None


    dataset_parts = []

    for channel_length, channel_width, interval in product(channel_lengths, channel_widths, intervals):
        print(f"l={channel_length} w={channel_width} i={interval}", end='  ', flush=True)
        if use_cached and outdir and (outdir / f"l-{channel_length}-w-{channel_width}-i-{interval}" / 'dataset.csv').exists():
            data = pd.read_csv(outdir / f"l-{channel_length}-w-{channel_width}-i-{interval}" / 'dataset.csv').set_index(['channel_length', 'channel_width', 'interval', 'simulation_id', 'pulse_id'])
        else:
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



def evaluation_fn(log_interval, channel_width, channel_length, fields, k_neighbors, reconstruction, outdir=None, evaluation_logger=None, suffix='', **kwargs):

    interval = int(np.round(np.exp(log_interval)))

    dataset = generate_dataset_batch(
        channel_lengths=[channel_length],
        channel_widths=[channel_width],
        intervals=[interval],
        outdir=outdir,

        **kwargs
    )
    entropy = get_entropy(
        dataset.reset_index(),
        fields=fields,
        reconstruction=reconstruction,
        k_neighbors=k_neighbors,
        outpath=outdir / f"l-{channel_length}-w-{channel_width}-i-{interval}" / f"entropies{suffix}.csv")
    
    assert len(entropy) == 1
    if evaluation_logger is not None:
        evaluation_logger.append(entropy)
    return entropy.iloc[-1]['bitrate_per_min']



def find_optimal_bitrate(
    expected_maximums, logstep, scan_points,
    channel_widths,
    channel_lengths,
    outdir=None,
    fields = 'c',
    k_neighbors = 25,
    reconstruction = False,
    processes=20,
    **kwargs
    ):
    
    suffix = f"{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"
    channel_wls = list(product(channel_widths, channel_lengths))

    if outdir:
        (outdir / suffix).mkdir(exist_ok=True, parents=True)

    scan_ranges = (np.exp(np.linspace(
            np.log(expected_maximums) - scan_points * logstep, 
            np.log(expected_maximums) + scan_points * logstep,
            2 * scan_points + 1).T) // 1).astype('int')

    nearest_pulses = pd.concat([
        generate_dataset_batch(
            channel_lengths=[channel_length],
            channel_widths=[channel_width],
            intervals=intervals,
            outdir=outdir,
            processes=processes,
            **kwargs
        ) for (channel_width, channel_length), intervals in zip(channel_wls, scan_ranges)
    ], ignore_index=True)

    fields_letter_to_fields = {
        'c': ['c+0'],
        'rl': ['l0', 'r0'],
        'cm': ['c+0', 'c-1'],
        'cp': ['c+0', 'c+1'],
        'cmp': ['c+0', 'c-1', 'c+1'],
    }
    # for fields in 'c',:#, 'rl', 'cm', 'cp', 'cmp':
    #     for k_neighbors in (25,):
    #         for reconstruction in (False,):
    print(f"Estimating entropy {suffix}")
    (outdir / suffix).mkdir(exist_ok=True, parents=True)

    entropies = get_entropy(nearest_pulses.reset_index(), fields=fields_letter_to_fields[fields], reconstruction=reconstruction, k_neighbors=k_neighbors)
    entropies.to_csv(outdir / suffix / f"entropies.csv")

    result = get_optimum_from_scan(entropies, field='bitrate_per_hour', required_wls=product(channel_widths, channel_lengths))

    result['channel_length_sqrt'] = np.sqrt(result.index.get_level_values('channel_length'))
    result['optimal_interval_sq'] = result['optimal_interval']**2
    result['max_bitrate'] = result['max_value'] / 60
    result['max_bitrate_sq'] = result['max_bitrate']**2
    result['max_bitrate_log'] = np.log(result['max_bitrate'])
    result['max_bitrate_inv'] = 1 / result['max_bitrate']

    fig, ax = plot_scan(entropies, x_field='interval', c_field='channel_length')#.reset_index().set_index(['channel_width', 'channel_length', ['bitrate_per_hour'].unstack('channel_width').plot('interval', 'bitrate_per_hour', marker='o')
    ax.plot(result['optimal_interval'], result['max_value'], color='k', marker='o')
    
    plt.savefig(outdir / 'partial_results.png')
    plt.savefig(outdir / 'partial_results.svg')


    if outdir:
        result.to_csv((outdir / suffix) / f'optimized_bitrate.csv')
    return result, entropies



