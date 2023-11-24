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
from scripts.defaults import TEMP_DIR, PARAMETERS_DEFAULT
from scripts.binary import activity_to_arrival_times, arrival_times_to_dataset


def make_binary_protocol(interval, n_slots, p=0.5, seed=0):
    random_generator = np.random.default_rng(seed)
    pulse_bits = random_generator.choice(2, size=n_slots, p=[1-p, p])
    pulse_bits[0] = 1

    locs_1, = np.nonzero(pulse_bits)

    pulse_times = interval * locs_1
    pulse_intervals = pulse_times[1:] - pulse_times[:-1]
    
    return pulse_times, pulse_intervals


def perform_single(interval, n_slots, duration, offset, n_margin, n_nearest, sim_dir, client, simulation_id):
    departure_times, pulse_intervals = make_binary_protocol(interval, n_slots, p=0.5, seed=19+simulation_id)
        
    protocol_file_path = make_protocol(
            pulse_intervals=list(pulse_intervals) + [1500],
            duration=duration,
            out_folder=sim_dir / f"sim-{simulation_id}",
        )

    result = client.run(
            parameters_json=PARAMETERS_DEFAULT,
            protocol_file_path=protocol_file_path,
            verbose=False,
            dir_name=f"sim-{simulation_id}/simulation",
            seed=simulation_id+19,
            states=False,
            activity=True,
        )
    # print('Result_loaded')
    arrival_times = activity_to_arrival_times(result.activity)
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
    channel_width=7,
    channel_length=300,
    duration=5,
    offset=None,
    v=1/3.6,
    n_margin=1,
    n_nearest=4, # how should this be determined?
    processes=None,
):
    # TEMP_DIR = "/tmp"

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
                n_margin=n_margin,
                n_nearest=n_nearest,
                sim_dir=sim_dir,
                client=client,
                simulation_id=simulation_id,
                )
            for simulation_id in range(n_simulations)
        ], processes=processes,
    ))

    # print('starmap completed')

    dataset['channel_width'] = channel_width
    dataset['channel_length'] = channel_length
    dataset['interval'] = interval

    rmtree(sim_dir)

    return dataset


def generate_dataset_batch(
        channel_lengths,
        channel_widths,
        intervals, 
        v=1/3.6,
        n_slots=100,
        n_simulations=15,
        duration=5,
        outpath=None,
        n_margin=5,
        n_nearest=1,
        append=False,
        flush_every_iteration=False,
        processes=None,
    ):
    if append and Path(outpath).exists():
        dataset_parts = [
            dataset_part for _, dataset_part in pd.read_csv(outpath).groupby(['channel_length', 'channel_width', 'interval']) #data.drop(columns=[col for col in data.columns if col.startswith('Unnamed')]) for _, data in pd.read_csv(outpath).groupby(['channel_length', 'channel_width', 'interval'], group_keys=False)
        ]
        dataset = pd.concat(dataset_parts, ignore_index=True)
        dataset = dataset.set_index(['channel_length', 'channel_width', 'interval', 'simulation_id', 'pulse_id'])
        # if outpath:
        #     data_all.to_csv(outpath)
    else:
        dataset_parts = []

    for channel_length, channel_width, interval in product(channel_lengths, channel_widths, intervals):
        print(f"l={channel_length} w={channel_width} i={interval}", end='', flush=True)
        data = generate_dataset(
            interval=interval,
            channel_width=channel_width,
            channel_length=channel_length,
            offset=channel_length / v,
            n_slots=n_slots,
            n_simulations=n_simulations,
            duration=duration,
            n_margin=n_margin,
            n_nearest=n_nearest,
            processes=processes,
        )
        data.index.name = 'pulse_id'
        dataset_parts.append(data.reset_index())
        if flush_every_iteration:
            dataset = pd.concat(dataset_parts, ignore_index=True).set_index(['channel_length', 'channel_width', 'interval', 'simulation_id', 'pulse_id'])
            if outpath:
                dataset.to_csv(outpath)
    if not flush_every_iteration:
        dataset = pd.concat(dataset_parts, ignore_index=True).set_index(['channel_length', 'channel_width', 'interval', 'simulation_id', 'pulse_id'])
        if outpath:
            dataset.to_csv(outpath)
    return dataset



    # fig.savefig(outdir / 'bitrates.png')


if __name__ == '__main__':
    nearest_pulses = pd.read_csv('../private/binary/approach2/data_all.csv')
    results = get_entropy(nearest_pulses, fields=['c'])
    plot_scan(
        results, 
        c_field='channel_width',
        x_field='interval',
        y_field='bitrate_per_hour',
    )
    plt.show()

