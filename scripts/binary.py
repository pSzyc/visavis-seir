import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pathlib import Path
from subplots_from_axsize import subplots_from_axsize
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt
from warnings import warn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) # in order to be able to import from scripts.py

from scripts.entropy_utils import conditional_entropy_discrete, conditional_entropy_discrete_reconstruction, conditional_entropy_discrete_bins_or_neighbors_pandas


def activity_to_arrival_times(activity, min_distance_between_peaks=6, end=-1):
   
    arrival_idxs, _ = find_peaks(activity[activity.columns[end]].to_numpy(), distance=min_distance_between_peaks)
    arrival_times = activity.index.get_level_values('seconds')[arrival_idxs]
    
    return arrival_times


def determine_offset(departure_times, arrival_times, interval, interval_after, n_slots, channel_length, v_approx=1/3.6, approximation_error=.3, fine_parameter=5):

    dt = interval / fine_parameter

    arrival_times_fft = np.exp(1j * arrival_times * (2 * np.pi / dt)).sum()
    departure_times_fft = np.exp(1j * departure_times * (2 * np.pi / dt)).sum()

    microoffset = (np.imag(np.log(arrival_times_fft / np.abs(arrival_times_fft))) / (2 * np.pi) * dt  + dt) % dt

    arrivals_binned = (arrival_times - microoffset + dt / 2) // dt

    slot_ids = pd.Series(np.arange((n_slots * interval + interval_after) // dt))
    timeline_departures = slot_ids.isin(departure_times // dt)
    timeline_arrivals = slot_ids.isin(arrivals_binned)

    approx_time = channel_length / v_approx

    scan_min = int((1 - approximation_error) * approx_time // dt)
    scan_max = int((1 +  approximation_error) * approx_time // dt) + 2
    cross_correlation = [(timeline_departures.shift(n) * timeline_arrivals).sum() for n in range(scan_min, scan_max)]
    plt.plot((np.arange(len(cross_correlation)) + scan_min) * interval / fine_parameter + microoffset, cross_correlation)

    offset = (np.argmax(cross_correlation) + scan_min) * interval / fine_parameter + microoffset

    return offset




def arrival_times_to_dataset(
    interval,
    departure_times,
    arrival_times,
    offset,
    n_nearest,
    n_margin=0,
):
    if len(arrival_times) < 2 * n_margin + 1:
        return pd.DataFrame([], columns=['x'] + [f'c{i:+d}' for i in range(-(n_nearest - 1), n_nearest)] + [f'l{i}' for i in reversed(range(n_nearest))] + [f'r{i}' for i in range(n_nearest)])

    is_pulse_s = []
    near_arrivals_s = []
    pulse_id_s = []
    # while t0 <= max(departure_times):   
    for pulse_id in range(0, len(departure_times)):
        t0 = pulse_id * interval
        
        # was it a pulse?
        is_pulse = t0 in departure_times

        # predicted arrival time
        t1 = t0 + offset

        # find predicted arrival time in actual arrival times
        idx = np.searchsorted(arrival_times, t1)

        # if there is not enough pulses before/after don't add to dataset
        if not n_nearest <= idx <= len(arrival_times) - n_nearest:
            continue

        # subtract predicted arrival time!
        near_arrivals = arrival_times[idx-n_nearest:idx+n_nearest] - t1

        is_pulse_s.append(is_pulse)
        near_arrivals_s.append(near_arrivals)
        pulse_id_s.append(pulse_id)



    if len(is_pulse_s) < 2 * n_margin + 1:
        return pd.DataFrame([], columns=['x'] + [f'c{i:+d}' for i in range(-(n_nearest - 1), n_nearest)] + [f'l{i}' for i in reversed(range(n_nearest))] + [f'r{i}' for i in range(n_nearest)])
    # drop margins to increase quality of samples
    is_pulse_s = np.array(is_pulse_s[n_margin:-n_margin] if n_margin else is_pulse_s)
    near_arrivals_s = np.array(near_arrivals_s[n_margin:-n_margin] if n_margin else near_arrivals_s)
    pulse_id_s = np.array(pulse_id_s[n_margin:-n_margin] if n_margin else pulse_id_s)

        # compute closest pulse
    nearest_arrival_idxs = np.argmin(np.abs(near_arrivals_s), axis=1, keepdims=True)
        
    dataset_part = pd.DataFrame({
        'pulse_id': pulse_id_s,
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
    }).set_index('pulse_id')
    return dataset_part


def get_entropy(dataset: pd.DataFrame, fields=['c'], reconstruction=False, k_neighbors=15, outpath=None):
    results = []
    get_conditional_entropy = conditional_entropy_discrete_reconstruction if reconstruction else conditional_entropy_discrete

    for (channel_width, channel_length, interval), data in dataset.groupby(['channel_width', 'channel_length', 'interval']):
        # cond_entropy = get_conditional_entropy(
        #         data['x'].to_numpy(),
        #         data[fields].to_numpy().reshape(-1, len(fields)),
        #         n_neighbors=k_neighbors,
        #     )

        cond_entropy = conditional_entropy_discrete_bins_or_neighbors_pandas(
            data,
            'x',
            fields,
            classes=[False, True],
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
            'bitrate_per_min': bitrate_per_min,
            'bitrate_per_hour': bitrate_per_hour,
        })
        
    results = pd.DataFrame(
        results, 
        columns=['channel_width', 'channel_length', 'interval', 'cond_entropy', 'efficiency', 'bitrate_per_min', 'bitrate_per_hour'],
    )
    if outpath:
        results.to_csv(outpath)
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
            **{
                'label': f'{c_val}',
                'color': f"C{c_it}",
                **kwargs,
            }
        )
        

    if x_field == 'interval' and y_field in ('bitrate_per_hour', 'efficiency'):
        ax.plot(
            x_vals,
            60 / x_vals if y_field == 'bitrate_per_hour' else np.ones(len(x_vals)),
            '-',
            lw=1,
            # alpha=0.3,
            color='grey',
            label=f'perfect',
        )
        
    ax.set_xlabel(x_field)
    ax.set_ylabel(
        'bitrate [bit/hour]' if y_field == 'bitrate_per_hour' else
        'transmission efficiency' if y_field == 'efficiency' else
        y_field
        )
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

