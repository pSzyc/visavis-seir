from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import generate_dataset_batch
from scripts.binary import get_entropy, get_optimal_bitrate


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4C' / 'approach3'
data_dir.mkdir(parents=True, exist_ok=True)
    
channel_widths = [3,4,5,6,7,8,9,10,11,12]
# channel_widths = [4,8,12]
# channel_widths = [5,6,7,10]
# channel_widths = [3]
# channel_widths = [9,11]

channel_lengths = [30,300]
# channel_lengths = [1700]

channel_wls = list(product(channel_widths, channel_lengths))


def find_optimal_bitrate(
    expected_maximums, logstep, scan_points,
    n_slots=100,
    n_simulations=100,
    outdir=None,
    fields = 'c',
    k_neighbors = 25,
    reconstruction = False,
    processes=20,
    ):

    if outdir:
        outdir.mkdir(exist_ok=True, parents=True)

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
            append=True,
            processes=processes,
            use_cached=True,
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
    suffix = f"{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"
    print(f"Estimating entropy {suffix}")
    (data_dir / suffix).mkdir(exist_ok=True, parents=True)

    entropies = get_entropy(nearest_pulses.reset_index(), fields=fields_letter_to_fields[fields], reconstruction=reconstruction, k_neighbors=k_neighbors)
    entropies.to_csv(data_dir / suffix / f"entropies.csv")

    result = get_optimal_bitrate(entropies, outdir=data_dir / suffix)

    return result

def get_expected_maximum(channel_width, channel_length):
    return 3.13 * np.sqrt(channel_length) + 81.7  + 0.6*(np.log10(channel_length)-.3)*(channel_width-6)**2

expected_maximums = get_expected_maximum(*np.array(channel_wls).T)
# interval_scan_steps = expected_maximums // 10
# interval_scan_centers = expected_maximums // 1
# scan_points = 3

result =  find_optimal_bitrate(expected_maximums, logstep=0.10, scan_points=11, outdir=data_dir / 'iteration1', k_neighbors=12, n_slots=200, n_simulations=20)
expected_maximums = result['optimal_interval']
result = find_optimal_bitrate(expected_maximums, logstep=0.06, scan_points=7, outdir=data_dir / 'iteration2', k_neighbors=18, n_slots=200, n_simulations=20)
expected_maximums = result['optimal_interval']
result = find_optimal_bitrate(expected_maximums, logstep=0.04, scan_points=5, outdir=data_dir / 'iteration3', k_neighbors=25, n_slots=500, n_simulations=60, processes=10)


