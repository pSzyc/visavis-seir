import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from shutil import rmtree

import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) # in order to be able to import from scripts.py

from scripts.client import VisAVisClient, _random_name
from scripts.make_protocol import make_protocol
from scripts.utils import compile_if_not_exists, starmap
from scripts.defaults import TEMP_DIR, PARAMETERS_DEFAULT, MOL_STATES_DEFAULT
from scripts.binary import activity_to_arrival_times, arrival_times_to_dataset, get_entropy


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
    n_slots, 
    duration, 
    offset, 
    n_margin, 
    n_nearest, 
    min_distance_between_peaks,
    sim_dir, 
    client, 
    simulation_id, 
    parameters=PARAMETERS_DEFAULT, 
    mol_states=MOL_STATES_DEFAULT, 
    p=0.5):
    departure_times, pulse_intervals = make_binary_protocol(interval, n_slots, p=p, seed=19+simulation_id)

    protocol_file_path = make_protocol(
            pulse_intervals=list(pulse_intervals) + [1500],
            duration=duration,
            out_folder=sim_dir / f"sim-{simulation_id}",
        )

    result = client.run(
            parameters_json=parameters,
            mol_states_json=mol_states,
            protocol_file_path=protocol_file_path,
            verbose=False,
            dir_name=f"sim-{simulation_id}/simulation",
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
    mol_states=MOL_STATES_DEFAULT,
    duration=5,
    offset=None,
    v=1/3.6,
    p=0.5,
    outdir=None,
    n_margin=4,
    n_nearest=4,
    min_distance_between_peaks=5,
    processes=None,
):

    # TEMP_DIR = "/tmp"
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    sim_dir = Path(f"{TEMP_DIR}/visavis_seir/binary/" + _random_name(12))
    visavis_bin = compile_if_not_exists(channel_width, channel_length)

    client = VisAVisClient(
        visavis_bin=visavis_bin,
        sim_root=sim_dir,
    )

    if offset is None:
        offset = channel_length / v

   
    dataset = pd.concat(starmap(
        perform_single,
        [
            dict(
                interval=interval,
                n_slots=n_slots,
                duration=duration,
                offset=offset,
                parameters=parameters,
                mol_states=mol_states,
                p=p,
                n_margin=n_margin,
                n_nearest=n_nearest,
                sim_dir=sim_dir,
                client=client,
                simulation_id=simulation_id,
                min_distance_between_peaks=min_distance_between_peaks,
                )
            for simulation_id in range(n_simulations)
        ], processes=processes,
    ))

    # print('starmap completed')

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
        v=1/3.6,
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
                offset=channel_length / v,
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
    
    # plt.plot(interval, entropy.iloc[-1]['bitrate_per_hour'], marker='o', color=f"C{channel_lengths.index(channel_length)}")
    assert len(entropy) == 1
    if evaluation_logger is not None:
        evaluation_logger.append(entropy)
    return entropy.iloc[-1]['bitrate_per_min']


