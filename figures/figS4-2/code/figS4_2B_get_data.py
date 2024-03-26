from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
from sklearn.linear_model import LinearRegression

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.defaults import PARAMETERS_DEFAULT
from scripts.propensities import simulate

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS4-2' / 'figS4-2B' / 'approach1'
data_dir.mkdir(exist_ok=True, parents=True)

channel_length = 300
channel_widths = [1,4,5,6,10,16,20]#[::-1]


species = ['i', 'e', 'r']
n_states_s = [1,2,6,10]

result_parts = []

for species, n_states in product(species, n_states_s):

    altered_parameter = f"{species}_subcompartments_count"

    (data_dir / altered_parameter / str(n_states)).mkdir(exist_ok=True, parents=True)


    parameters = PARAMETERS_DEFAULT.copy()
    parameters[f'{species}_subcompartments_count'] = n_states
    parameters[f'{species}_forward_rate'] = n_states * PARAMETERS_DEFAULT[f'{species}_forward_rate']  / PARAMETERS_DEFAULT[f'{species}_subcompartments_count']

    v = 1.25  / (parameters['e_subcompartments_count'] / parameters['e_forward_rate'] + 0.5 / parameters['c_rate'])


    propensities = simulate(
        n_sim=30000,
        channel_widths=channel_widths,
        results_file=data_dir / altered_parameter / str(n_states) / 'fig2C--propensities.csv',
        channel_length=channel_length,
        n_workers=20,
        interval_after=int(2.5 * channel_length / v),
        parameters=parameters,
        v=v,
        per_width_kwargs = {
            w: {
                'front_direction_minimal_distance': min(max(w - 1, 1), 5),
                'min_peak_height': 0.03 / w,
            } for w in channel_widths
        },
        # use_cached=True,
        # plot_results=True,
        save_iterations=False,
        ).reset_index()

    print(propensities)

    lreg = LinearRegression().fit(propensities[['channel_width']], propensities[['l_spawning']])
    a_sp, b_sp = lreg.coef_[0,0], lreg.intercept_[0]

    propensities_cropped = propensities[propensities['channel_width'].le(6) & propensities['l_failure'].gt(0)]
    lreg_failure = LinearRegression().fit(propensities_cropped[['channel_width']], np.log(propensities_cropped[['l_failure']]))
    a_fail, b_fail = lreg_failure.coef_[0,0], lreg_failure.intercept_[0]

    result_part = {
        'altered_parameter': altered_parameter,
        'n_states': n_states,
        'a_spawning': a_sp,
        'b_spawning': b_sp,
        'a_failure': a_fail,
        'b_failure': b_fail,
    }
    result_parts.append(result_part)

result = pd.DataFrame(result_parts).set_index(['altered_parameter', 'n_states'])
result.to_csv(data_dir / 'fit_coefficients.csv')

