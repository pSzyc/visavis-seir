from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from tqdm import tqdm

#import matplotlib.pyplot as plt
#import matplotlib.ticker as mticker
# from subplots_from_axsize import subplots_from_axsize

print(0, flush=True)

#import sys
#sys.path.insert(0, '..') # in order to be able to import from scripts.py

print(1, flush=True)

from scripts.client import VisAVisClient
from scripts.make_protocol import make_protocol
from scripts.entropy_utils import conditional_entropy_discrete
from scripts.plot_result import plot_result

PARAMETERS_DEFAULT = {
  "c_rate": 1,
  "e_forward_rate": 1,
  "i_forward_rate": 1,
  "r_forward_rate": 0.0667,
  "e_subcompartments_count": 4,
  "i_subcompartments_count": 2,
  "r_subcompartments_count": 4
}


def make_binary_protocol(interval, n_slots, p=0.5):
    pulse_bits = np.random.choice(2, size=n_slots, p=[1-p, p])
    pulse_bits[0] = 1

    locs_1, = np.nonzero(pulse_bits)

    pulse_times = interval * locs_1
    pulse_intervals = pulse_times[1:] - pulse_times[:-1]
    
    return pulse_times, pulse_intervals


def results_to_arrival_times(
    result,
):
    assert result.states

    h_max = result.states['h'].max()
    data = result.states[result.states['h'] == h_max].copy()
    
    data['act'] = 1.0 * ((data['E'] > 0) | (data['I'] > 0))
    data_grouped = data.groupby(['seconds', 'h'])['act'].mean()
    
    output_series = data_grouped[:, h_max]
    arrival_idxs, _ = find_peaks(output_series, distance=5)
    arrival_times = output_series.index[arrival_idxs]
    
    return arrival_times


def construct_dataset(
    interval,
    departure_times,
    arrival_times,
    offset=720, # how should this be determined?
    n_nearest=1,
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

    xs = np.array(xs)
    yss = np.array(yss)
    return xs, yss


def generate_dataset(
    interval,
    n_bits,
    n_simulations,
    channel_width=7,
    channel_length=300,
    duration=5,
    n_margin=5,
    n_nearest=1,
    offset=720, # how should this be determined?
):
    client = VisAVisClient()
    
    data_parts = []
    for simulation_id in tqdm(range(n_simulations)):
        departure_times, pulse_intervals = make_binary_protocol(interval, n_bits, p=0.5)
        
        protocol_file_path = make_protocol(
            pulse_intervals=list(pulse_intervals) + [1500],
            duration=duration,
            out_folder='/tmp/',
        )

        result = client.run(
            channel_length=channel_length,
            channel_width=channel_width,
            parameters_json=PARAMETERS_DEFAULT,
            protocol_file_path=protocol_file_path,
            verbose=False,
        )
        
        arrival_times = results_to_arrival_times(result)
        
        xs, yss = construct_dataset(
            interval=interval,
            departure_times=departure_times,
            arrival_times=arrival_times,
            n_nearest=n_nearest, 
            offset=offset,
        )

        assert n_nearest == 1 # TODO rewrite the below to handle not only the nearest pulse
        # compute closest pulse
        yss_abs_argmins = np.argmin(np.abs(yss), axis=1, keepdims=True)
        cs = np.take_along_axis(yss, yss_abs_argmins, axis=1)[:, 0]
        
        # drop margins to increase quality of samples
        xs = xs[n_margin:-n_margin]
        cs = cs[n_margin:-n_margin]
        yss = yss[n_margin:-n_margin]

        data_part = pd.DataFrame({'x': xs, 'c': cs})
        for i in range(n_nearest):
            data_part[f'l{i}'] = yss[:, n_nearest-i-1]
            data_part[f'r{i}'] = yss[:, n_nearest+i]

        data_part['simulation_id'] = simulation_id
        data_parts.append(data_part)
        
    return pd.concat(data_parts)


if __name__ == '__main__':
    print(2, flush=True)
    outdir = Path('../private/binary_attempt_2023-09-26')
    outdir.mkdir(parents=True, exist_ok=True)
    
    data_parts = []

    for channel_length in [30, 45, 70, 100, 130, 200, 250, 300, 500, 700, 1000]:
        for channel_width in [6]:
            for interval in [60, 110, 130, 180]:
                data = generate_dataset(
                    interval=interval,
                    channel_width=channel_width,
                    channel_length=channel_length,
                    offset=3.6*channel_length,
                    n_bits=100,
                    n_simulations=15,
                    duration=5,
                )

                data['channel_width'] = channel_width
                data['channel_length'] = channel_length
                data['interval'] = interval
                
                data_parts.append(data)
                data_all = pd.concat(data_parts)
                data_all.to_csv(outdir / 'data_all-Lscan.csv')
