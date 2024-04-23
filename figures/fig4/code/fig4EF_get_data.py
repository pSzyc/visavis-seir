from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import generate_dataset_batch
from scripts.binary import get_entropy, plot_scan
from scripts.optimizer import get_optimum_from_scan
from scripts.formula import get_expected_maximum_for_defaults


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4EF' / 'approach7'
data_dir.mkdir(parents=True, exist_ok=True)
    
channel_widths_s = (
    [6,5,4,3],
    [6,7,8,9,10,11,12],
    )

channel_lengths = [100,300,1000] #,1000


fields_letter_to_fields = {
    'c': ['c+0'],
    'rl': ['l0', 'r0'],
    'cm': ['c+0', 'c-1'],
    'cp': ['c+0', 'c+1'],
    'cmp': ['c+0', 'c-1', 'c+1'],
}


def find_optimal_bitrate(
    expected_maximums, logstep, scan_points,
    channel_lengths,
    channel_width,
    n_slots=100,
    n_simulations=100,
    outdir=None,
    fields='c',
    k_neighbors=25,
    reconstruction=False,
    processes=20,
    **kwargs
    ):
    
    suffix = f"{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"

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
            n_simulations=n_simulations,
            n_slots=n_slots,
            n_margin=4,
            n_nearest=4,
            duration=1,
            processes=processes,
            **kwargs
        ) for channel_length, intervals in zip(channel_lengths, scan_ranges)
    ], ignore_index=True)

 
    print(f"Estimating entropy {suffix}")
    (outdir / suffix).mkdir(exist_ok=True, parents=True)

    entropies = get_entropy(nearest_pulses.reset_index(), fields=fields_letter_to_fields[fields], reconstruction=reconstruction, k_neighbors=k_neighbors)
    entropies.to_csv(outdir / suffix / f"entropies.csv")

    result = get_optimum_from_scan(entropies, field='bitrate_per_hour')

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



fields = 'c'
k_neighbors = 25
reconstruction = False
suffix = f"{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"

result_parts = []
entropies_parts = []

for channel_widths in channel_widths_s:
    expected_maximums = get_expected_maximum_for_defaults(np.array(channel_lengths))
    for channel_width in channel_widths:
        result, entropies = find_optimal_bitrate(
            expected_maximums, logstep=0.03, scan_points=5, 
            channel_lengths=channel_lengths,
            channel_width=channel_width,
            outdir=data_dir / f'w-{channel_width}', 
            k_neighbors=k_neighbors, 
            reconstruction=reconstruction,
            fields=fields,
            n_slots=500, 
            n_simulations=100, #60 
            processes=10,
            use_cached=True,
            )
        expected_maximums = result['optimal_interval']
        result_parts.append(result)
        entropies_parts.append(entropies)


result = pd.concat(result_parts).reset_index().drop_duplicates().set_index(['channel_width', 'channel_length']).sort_index()
result.to_csv((data_dir / suffix) / f'optimized_bitrate-{suffix}.csv')

entropies = pd.concat(entropies_parts)
entropies.to_csv(data_dir / f'entropies-{suffix}.csv')

