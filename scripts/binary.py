from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) # in order to be able to import from scripts.py

from scripts.client import VisAVisClient, _random_name
from scripts.make_protocol import make_protocol
from scripts.utils import compile_if_not_exists, starmap
from scripts.defaults import TEMP_DIR


PARAMETERS_DEFAULT = {
  "c_rate": 1,
  "e_incr": 1,
  "i_incr": 1,
  "r_incr": 0.0667
}


def make_binary_protocol(interval, n_slots, p=0.5):
    pulse_bits = np.random.choice(2, size=n_slots, p=[1-p, p])
    pulse_bits[0] = 1

    locs_1, = np.nonzero(pulse_bits)

    pulse_times = interval * locs_1
    pulse_intervals = pulse_times[1:] - pulse_times[:-1]
    
    return pulse_times, pulse_intervals


def results_to_arrival_times(result):

    data = result.states
    h_max = data['h'].max()

    data['act'] = 1.0 * ((data['E'] > 0) | (data['I'] > 0))
    data_grouped = data.groupby(['seconds', 'h'])['act'].mean()
    
    output_series = data_grouped[:, h_max]
    arrival_idxs, _ = find_peaks(output_series, distance=5)
    arrival_times = output_series.index[arrival_idxs]
    
    return arrival_times


def arrival_times_to_dataset(
    interval,
    departure_times,
    arrival_times,
    offset, 
    n_nearest,
    n_margin=0,
):
    t0 = 0
    xs = []
    yss = []
    while t0 <= max(departure_times):
        t0 += interval
        
        # was it a pulse?
        x = t0 in departure_times

        # predicted arrival time
        t1 = t0 + offset

        # find predicted arrival time in actual arrival times
        idx = np.searchsorted(arrival_times, t1)

        # if there is not enough pulses before/after don't add to dataset
        if idx < n_nearest or idx > len(arrival_times) - n_nearest:
            continue

        # subtract predicted arrival time!
        ys = arrival_times[idx-n_nearest:idx+n_nearest] - t1

        xs.append(x)
        yss.append(ys)

        # drop margins to increase quality of samples
    xs = np.array(xs[n_margin:-n_margin])
    yss = np.array(yss[n_margin:-n_margin])

        # compute closest pulse
    yss_abs_argmins = np.argmin(np.abs(yss), axis=1, keepdims=True)
    cs = np.take_along_axis(yss, yss_abs_argmins, axis=1)[:, 0]
        
    data_part = pd.DataFrame({'x': xs, 'c': cs})
    for i in range(n_nearest):
        data_part[f'l{i}'] = yss[:, n_nearest-i-1]
        data_part[f'r{i}'] = yss[:, n_nearest+i]

    return data_part


def perform_single(interval, n_slots, duration, offset, n_margin, n_nearest, sim_dir, client, simulation_id):
    departure_times, pulse_intervals = make_binary_protocol(interval, n_slots, p=0.5)
        
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
        )
    # print('Result_loaded')
    arrival_times = results_to_arrival_times(result)
    # print('Arrivals_computed')
        
    data_part = arrival_times_to_dataset(
            interval=interval,
            departure_times=departure_times,
            arrival_times=arrival_times,
            n_nearest=n_nearest, 
            n_margin=n_margin,
            offset=offset,
        )
    # print('Data set constructed')

    data_part['simulation_id'] = simulation_id

    return data_part


def generate_dataset(
    interval,
    n_slots,
    n_simulations,
    channel_width=7,
    channel_length=300,
    duration=5,
    offset=None,
    n_margin=1,
    n_nearest=4, # how should this be determined?
    processes=20,
):
    # TEMP_DIR = "/tmp"

    sim_dir = Path(f"{TEMP_DIR}/visavis_seir/binary/" + _random_name(12))
    visavis_bin = compile_if_not_exists(channel_width, channel_length)

    client = VisAVisClient(
        visavis_bin=visavis_bin,
        sim_root=sim_dir,
    )

    if offset is None:
        offset = channel_length * 3.6
   
    data = pd.concat(starmap(
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

    data['channel_width'] = channel_width
    data['channel_length'] = channel_length
    data['interval'] = interval

    sim_dir.unlink()

    return data


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
        processes=20,
    ):
    if append and Path(outpath).exists():
        data_parts = [
            data.drop(columns=[col for col in data.columns if col.startswith('Unnamed')]) for _, data in pd.read_csv(outpath).groupby(['channel_length', 'channel_width', 'interval'], group_keys=False)
        ]
        data_all = pd.concat(data_parts, ignore_index=True)
        data_all = data_all.set_index(['channel_length', 'channel_width', 'interval', 'simulation_id', 'pulse_id'])
        if outpath:
            data_all.to_csv(outpath)
    else:
        data_parts = []

    for channel_length in channel_lengths:
        for channel_width in channel_widths:
            for interval in intervals:
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
                
                data_parts.append(data.reset_index())
                data_all = pd.concat(data_parts, ignore_index=True)
                data_all = data_all.set_index(['channel_length', 'channel_width', 'interval', 'simulation_id', 'pulse_id'])
                if outpath:
                    data_all.to_csv(outpath)
    return data_all


if __name__ == '__main__':
    outdir = Path('../private/binary/sample')
    outdir.mkdir(parents=True, exist_ok=True)

    # channel_widths = list(range(4,10)) + list(range(10,21,2))
    channel_widths = [6]#list(range(3,10,2)) + list(range(10,21,3))
    
    generate_dataset_batch(
        channel_lengths=[30, 100],
        channel_widths=channel_widths,
        intervals=[30,80,130],#[60,70,80,90,100],#list(range(110, 181, 5)), #[60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130, 150, 180, 220, 260, 300, 400], 
        outpath=outdir / 'nearest_pulses.csv',
        n_simulations=2,
        n_slots=500,
        n_margin=4,
        n_nearest=4,
        append=True,
        processes=20,
    )

