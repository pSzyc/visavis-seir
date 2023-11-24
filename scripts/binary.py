import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path
from subplots_from_axsize import subplots_from_axsize
from matplotlib.ticker import MultipleLocator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) # in order to be able to import from scripts.py

from scripts.entropy_utils import conditional_entropy_discrete, conditional_entropy_discrete_reconstruction


# def results_to_arrival_times(result):

#     data = result.states
#     h_max = data['h'].max()

#     data['act'] = 1.0 * ((data['E'] > 0) | (data['I'] > 0))
#     data_grouped = data.groupby(['seconds', 'h'])['act'].mean()
    
#     output_series = data_grouped[:, h_max]
#     arrival_idxs, _ = find_peaks(output_series, distance=5)
#     arrival_times = output_series.index[arrival_idxs]
    
#     return arrival_times

def activity_to_arrival_times(activity):
   
    arrival_idxs, _ = find_peaks(activity[activity.columns[-1]].to_numpy(), distance=5)
    arrival_times = activity.index.get_level_values('seconds')[arrival_idxs]
    
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
    is_pulse_s = []
    near_arrivals_s = []
    while t0 <= max(departure_times):
        t0 += interval
        
        # was it a pulse?
        is_pulse = t0 in departure_times

        # predicted arrival time
        t1 = t0 + offset

        # find predicted arrival time in actual arrival times
        idx = np.searchsorted(arrival_times, t1)

        # if there is not enough pulses before/after don't add to dataset
        if idx < n_nearest or idx > len(arrival_times) - n_nearest:
            continue

        # subtract predicted arrival time!
        near_arrivals = arrival_times[idx-n_nearest:idx+n_nearest] - t1

        is_pulse_s.append(is_pulse)
        near_arrivals_s.append(near_arrivals)

        # drop margins to increase quality of samples

    if len(is_pulse_s) < 2 * n_margin + 1:
        return pd.DataFrame([], columns=['x'] + [f'c{i:+d}' for i in range(-(n_nearest - 1), n_nearest)] + [f'l{i}' for i in reversed(range(n_nearest))] + [f'r{i}' for i in range(n_nearest)])
    is_pulse_s = np.array(is_pulse_s[n_margin:-n_margin])
    near_arrivals_s = np.array(near_arrivals_s[n_margin:-n_margin])

        # compute closest pulse
    nearest_arrival_idxs = np.argmin(np.abs(near_arrivals_s), axis=1, keepdims=True)
        
    dataset_part = pd.DataFrame({
        'x': is_pulse_s,
        **{
            f'c{i:+d}':  np.take_along_axis(near_arrivals_s, nearest_arrival_idxs + i, axis=1)[:, 0]
            for i in range(-(n_nearest - 1), n_nearest)
        },
        **{
            f'l{i}': near_arrivals_s[:, n_nearest-i-1]
            for i in reversed(range(n_nearest))
        },
        **{
            f'r{i}': near_arrivals_s[:, n_nearest+i]
            for i in range(n_nearest)
        }
    })
    return dataset_part


def get_entropy(dataset: pd.DataFrame, fields=['c'], reconstruction=False, k_neighbors=15):
    results = []
    get_conditional_entropy = conditional_entropy_discrete_reconstruction if reconstruction else conditional_entropy_discrete

    for (channel_width, channel_length, interval), data in dataset.groupby(['channel_width', 'channel_length', 'interval']):
        cond_entropy = get_conditional_entropy(
                data['x'].to_numpy(),
                data[fields].to_numpy().reshape(-1, len(fields)),
                n_neighbors=k_neighbors,
            )
        
        mi_slot = 1 - cond_entropy
        bitrate_per_min = mi_slot / interval # seconds in simulations are minutes in reality
        bitrate_per_hour = bitrate_per_min * 60
        
        results.append({
            'channel_width': channel_width,
            'channel_length': channel_length,
            'interval': interval,
            'cond_entropy': cond_entropy,
            'efficiency': mi_slot,
            'bitrate_per_hour': bitrate_per_hour,
        })
        
    results = pd.DataFrame(results)
    return results


def plot_scan(results, x_field, c_field, y_field='bitrate_per_hour', ax=None, fmt='-o', **kwargs):
    if ax == None:
        fig, ax = subplots_from_axsize(1, 1, (5, 4), left=0.7)
    else:
        fig = ax.get_figure()
    x_vals = np.unique(results[x_field])


    for c_it, (c_val, results_h) in enumerate(results.groupby(c_field)):
        ax.plot(
            results_h[x_field],
            results_h[y_field],
            fmt,
            color=f"C{c_it}",
            label=f'{c_val}',
            **kwargs,
        )
        

    if x_field == 'interval' and y_field in ('bitrate_per_hour', 'efficiency'):
        ax.plot(
            x_vals,
            60 / x_vals if y_field == 'bitrate_per_hour' else np.ones(len(x_vals)),
            ':',
            color='grey',
            label=f'perfect',
        )
        
    ax.set_xlabel(x_field)
    ax.set_ylabel('bitrate [bit/hour]')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


    if x_field == 'channel_length':
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        
    if x_field == 'interval':
        ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_minor_locator(MultipleLocator(.1))
    ax.grid(which='both', ls=':')

    ax.legend(title=c_field)

    return fig, ax


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

