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


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig6' / 'fig6C' / 'states'/ 'approach3'
data_dir.mkdir(parents=True, exist_ok=True)
    
channel_widths = [6]
# channel_lengths = [30, 300]
channel_lengths = [1000]
channel_lengths = [30,100,300,1000]
altered_parameters = ['e_subcompartments_count', 'i_subcompartments_count', 'r_subcompartments_count']


channel_wls = list(product(channel_widths, channel_lengths))


for altered_parameter in altered_parameters:
    for param_values in (
        range(PARAMETERS_DEFAULT[altered_parameter],0,-1),
        range(PARAMETERS_DEFAULT[altered_parameter],11,1),
    ):

        expected_maximums = get_expected_maximum_for_defaults(np.array([l for w,l in channel_wls]))

        for param_value in param_values:
            
            print(f"{altered_parameter} = {param_value}")
            corresponding_rate = f"{altered_parameter[0]}_forward_rate"

            parameters = PARAMETERS_DEFAULT.copy()
            parameters.update({
                altered_parameter: param_value,
                corresponding_rate: PARAMETERS_DEFAULT[corresponding_rate] * param_value / PARAMETERS_DEFAULT[altered_parameter]
            })

            result, entropies = find_optimal_bitrate(
                expected_maximums, logstep=0.04, scan_points=10, 
                parameters=parameters,
                channel_widths=channel_widths,
                channel_lengths=channel_lengths,
                outdir=data_dir / altered_parameter / f'{param_value:d}', 
                k_neighbors=25, 
                n_slots=500, 
                n_simulations=20,#60, 
                processes=10,
                use_cached=True,
                )
            expected_maximums = result['optimal_interval']


