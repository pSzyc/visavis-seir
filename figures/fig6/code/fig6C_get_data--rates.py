from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import find_optimal_bitrate
from scripts.formula import get_expected_maximum_for_defaults
from scripts.defaults import PARAMETERS_DEFAULT


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig6' / 'fig6C' / 'rates' / 'approach3'
data_dir.mkdir(parents=True, exist_ok=True)
    
channel_widths = [6]
channel_lengths = [30]#, 100, 300, 1000]
channel_lengths = [30, 100, 300, 1000]
altered_parameters = ['c_rate', 'e_forward_rate',  'i_forward_rate', 'r_forward_rate']
fold_changes_s = (
    np.exp(np.linspace(0, 1, 11)),
    np.exp(np.linspace(0, -1, 11)),
)


channel_wls = list(product(channel_widths, channel_lengths))


for altered_parameter in altered_parameters:
    for fold_changes in fold_changes_s:
        expected_maximums = get_expected_maximum_for_defaults(np.array([l for w,l in channel_wls]))
        for fold_change in fold_changes:
            
            print(f"{altered_parameter} * {fold_change}")
            parameters = PARAMETERS_DEFAULT.copy()
            parameters.update({
                altered_parameter: fold_change * parameters[altered_parameter]
            })

            result, entropies = find_optimal_bitrate(
                expected_maximums, logstep=0.04, scan_points=5, 
                channel_widths=channel_widths,
                channel_lengths=channel_lengths,
                parameters=parameters,
                outdir=data_dir / altered_parameter / f'{fold_change:.3f}', 
                k_neighbors=25, 
                n_slots=500, 
                n_simulations=20, #60 
                processes=10,
                use_cached=True,
                )
            expected_maximums = result['optimal_interval'].mask(result['optimal_interval'].isna(), expected_maximums)
        

