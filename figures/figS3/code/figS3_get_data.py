# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
 

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.propensities import get_propensities_batch

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3' / 'approach8' 
data_dir.mkdir(exist_ok=True, parents=True)

channel_widths = (list(range(1,10,1)) + list(range(10,21,2)))#[::-1]
channel_lengths = [300]

propensities = get_propensities_batch(
    channel_lengths=channel_lengths,
    channel_widths=channel_widths,
    lattice_top_edge_aperiodic=True,
    n_simulations=30000,
    interval_after=2660,
    results_file=data_dir / 'figS3--propensities.csv',
    n_workers=32,
    per_width_kwargs = {
        w: {
            'front_direction_minimal_distance': min(max(w - 1, 1), 5),
            'min_peak_height': 0.03 / w,
        } for w in channel_widths
    },
    plot_results=True,
    lazy=False,
    use_cached=True,
).reset_index()

for channel_length in channel_lengths:

    propensities_selected = propensities[propensities['channel_length'] == channel_length]

    propensities_cropped_for_spawning = propensities_selected[propensities_selected['l_spawning'] > 0.3 * propensities_selected['l_failure']]
    lreg = LinearRegression().fit(propensities_cropped_for_spawning[['channel_width']].to_numpy(), propensities_cropped_for_spawning[['l_spawning']].to_numpy())
    a_sp, b_sp = lreg.coef_[0,0], lreg.intercept_[0]

    propensities_cropped_for_failure = propensities_selected[(propensities_selected['channel_width'] <= 6) & (propensities_selected['l_failure'] > 0.3 * propensities_selected['l_spawning'])]
    lreg_failure = LinearRegression().fit(propensities_cropped_for_failure[['channel_width']].to_numpy(), np.log(propensities_cropped_for_failure[['l_failure']].to_numpy()))
    a_fail, b_fail = lreg_failure.coef_[0,0], lreg_failure.intercept_[0]

    coefs = pd.DataFrame({'spawning': {'a': a_sp, 'b': b_sp}, 'failure': {'a': a_fail, 'b': b_fail}})
    coefs.index.name = 'coefficient'
    coefs.to_csv(data_dir / f'coefs--l-{channel_length}.csv')

    print(coefs.loc['b'] / coefs.loc['a'])
