from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import generate_dataset_batch
from scripts.binary import get_entropy, get_optimal_bitrate
from scripts.defaults import PARAMETERS_DEFAULT


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4D' / 'approach1'
data_dir.mkdir(parents=True, exist_ok=True)
    
channel_widths = [6]
channel_lengths = [30, 300]
fold_changes = np.exp(np.linspace(0, 1, 11))

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

    result = get_optimal_bitrate(entropies, outdir=outdir / suffix)

    return result

def get_expected_maximum(channel_width, channel_length):
    return 3.13 * np.sqrt(channel_length) + 81.7  

expected_maximums = get_expected_maximum(*np.array(channel_wls).T)

# print(expected_maximums)

altered_parameter = 'i_forward_rate'
for fold_change in fold_changes:
    
    parameters = PARAMETERS_DEFAULT.copy()
    parameters.update({
        altered_parameter: fold_change * parameters[altered_parameter]
    })
    v = 1.25  / (PARAMETERS_DEFAULT['e_subcompartments_count'] / parameters['e_forward_rate'] + 0.5 / parameters['c_rate'])

    result = find_optimal_bitrate(
        expected_maximums, logstep=0.04, scan_points=5, 
        parameters=parameters,
        v=v,
        outdir=data_dir / altered_parameter / f'{fold_change:.3f}', 
        k_neighbors=25, 
        n_slots=500, 
        # n_slots=100, 
        n_simulations=60, 
        processes=10,
        # use_cached=True,
        )
    expected_maximums = result['optimal_interval']


