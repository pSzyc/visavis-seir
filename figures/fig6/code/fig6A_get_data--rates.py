# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
from matplotlib import pyplot as plt

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_periodic import generate_dataset_batch
from scripts.defaults import PARAMETERS_DEFAULT
from scripts.optimizer import get_optimum_from_scan
from scripts.formula import get_expected_maximum_for_defaults

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig6' / 'fig6A' / 'rates' / 'approach3'
data_dir.mkdir(parents=True, exist_ok=True)
    
channel_widths = [6]
channel_lengths = [300]
altered_parameters = ['c_rate', 'e_forward_rate',  'i_forward_rate', 'r_forward_rate']
# fold_changes = np.exp(np.linspace(0, 1, 11))
fold_changes_s = (
    np.exp(np.linspace(0, 1, 11)),
    np.exp(np.linspace(0, -1, 11)),
)



channel_wls = list(product(channel_widths, channel_lengths))



def find_optimal_bitrate(
    expected_maximums, logstep, scan_points,
    n_pulses=500,
    n_simulations=100,
    outdir=None,
    **kwargs
    ):
    

    if outdir:
        outdir.mkdir(exist_ok=True, parents=True)

    scan_ranges = (np.exp(np.linspace(
            np.log(expected_maximums) - scan_points * logstep, 
            np.log(expected_maximums) + scan_points * logstep,
            2 * scan_points + 1).T) // 1).astype('int')

    arrival_times = pd.concat([
        generate_dataset_batch(
            channel_lengths=[channel_length],
            channel_widths=[channel_width],
            intervals=intervals,
            outdir=outdir,
            n_simulations=n_simulations,
            n_pulses=n_pulses,
            **kwargs
        ) for (channel_width, channel_length), intervals in zip(channel_wls, scan_ranges)
    ], ignore_index=True)

    pulse_counts = arrival_times.groupby(['channel_width', 'channel_length', 'interval']).size() 

    frequencies = pulse_counts / (pulse_counts.index.get_level_values('interval') * n_simulations * n_pulses)
    frequencies.name = 'pulse_frequency_at_end'

    result = get_optimum_from_scan(frequencies.to_frame(), field='pulse_frequency_at_end')

    frequencies.to_csv(outdir / 'frequencies.csv')
    result.to_csv(outdir / 'optimized_frequencies.csv')
    ax = frequencies.reset_index().plot('interval', 'pulse_frequency_at_end', marker='o')
    ax.plot(result['optimal_interval'], result['max_value'], color='k', marker='o')
    
    plt.savefig(outdir / 'partial_results.png')
    plt.savefig(outdir / 'partial_results.svg')

    plt.close()

    return result


result_parts = []

for altered_parameter in altered_parameters:
    for fold_changes in fold_changes_s:
        expected_maximums = get_expected_maximum_for_defaults(np.array([l for w,l in channel_wls]))
        for fold_change in fold_changes:
            print(f"{altered_parameter} * {fold_change:.3f}")
            
            parameters = PARAMETERS_DEFAULT.copy()
            parameters.update({
                altered_parameter: fold_change * parameters[altered_parameter]
            })
            # v = 1.25  / (PARAMETERS_DEFAULT['e_subcompartments_count'] / parameters['e_forward_rate'] + 0.5 / parameters['c_rate'])

            result = find_optimal_bitrate(
                expected_maximums, logstep=0.03, scan_points=5, 
                parameters=parameters,
                outdir=data_dir / altered_parameter / f'{fold_change:.3f}', 
                n_pulses=500, 
                n_simulations=100, #60 
                processes=10,
                use_cached='always',
                )
            expected_maximums = result['optimal_interval']
            result['altered_parameter'] = altered_parameter
            result['fold_change'] = fold_change
            result_parts.append(result)


result = pd.concat(result_parts).reset_index().drop_duplicates().set_index(['channel_width', 'channel_length', 'altered_parameter', 'fold_change']).sort_index()
result.to_csv(data_dir / 'optimized_frequencies.csv')


