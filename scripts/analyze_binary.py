import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from subplots_from_axsize import subplots_from_axsize
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) # in order to be able to import from scripts.py

from scripts.entropy_utils import conditional_entropy_discrete, conditional_entropy_discrete_reconstruction


def get_entropy(data_all: pd.DataFrame, fields=['c'], reconstruction=False, k_neighbors=15):
    results = []

    for (channel_width, channel_length, interval), data in data_all.groupby(['channel_width', 'channel_length', 'interval']):
        cond_entropy = min([
            (conditional_entropy_discrete_reconstruction if reconstruction else conditional_entropy_discrete)(
                data['x'].to_numpy(),
                data[fields].to_numpy().reshape(-1, len(fields)),
                n_neighbors=k,
            )
            for k in [k_neighbors]# np.arange(10, 51, 5)
        ])
        
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
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(100))
        
    if x_field == 'interval':
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(.1))
    ax.grid(which='both', ls=':')

    ax.legend(title=c_field)

    return fig, ax

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

