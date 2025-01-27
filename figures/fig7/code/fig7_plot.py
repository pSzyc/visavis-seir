# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
 

from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize
from itertools import product
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.style import *
from scripts.defaults import PARAMETERS_DEFAULT


def get_corresponding_states(param):
    if param == 'c_rate':
        return 1
    else:
        return PARAMETERS_DEFAULT[f'{param[0]}_subcompartments_count']

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig7' / 'fig7' / 'approach5'
data_dir_rates = Path(__file__).parent.parent.parent.parent  / 'data' / 'fig5' /  'fig5A' / 'approach5'
panel_dir = Path(__file__).parent.parent / 'panels'
panel_dir.mkdir(parents=True, exist_ok=True)


channel_length = 300

coefficients_rates = pd.read_csv(data_dir_rates / 'fit_coefficients.csv').set_index(['altered_parameter', 'fold_change'])

coefficients_rates['parameter_value'] = [PARAMETERS_DEFAULT[altered_parameter] * fold_change for altered_parameter, fold_change in coefficients_rates.index]
coefficients_rates['n_states'] = list(map(get_corresponding_states, coefficients_rates.index.get_level_values('altered_parameter')))
coefficients_rates['tau'] = coefficients_rates['n_states'] / coefficients_rates['parameter_value']


fig, axs = subplots_from_axsize(1, 2, (2.7, 2), wspace=.35, right=.4, sharey=True)

for ax, (altered_parameter, fold_changes) in zip(axs, [
    ('r_forward_rate', [6/5, 6/6, 6/7, 6/8, 6/9]),
    ('e_forward_rate', [.7, .9, 1., 1.1, 1.3]),
    # ('e_forward_rate', [4/2, 4/3, 4/4, 4/5, 4/6]),
]):
    parameter_letter = altered_parameter[0]
    corresponding_states = f"{parameter_letter}_subcompartments_count"

    coefs = coefficients_rates.loc[altered_parameter].reset_index().set_index('tau')

    probabilities = pd.concat([
            pd.read_csv(data_dir / f'{parameter_letter}_rate' / f"{fold_change:.3f}" / 'probabilities.csv').set_index(['channel_length', 'channel_width', 'interval'])
            for fold_change in fold_changes
        ], 
        names=['fold_change'],
        keys=fold_changes,
    )

    probabilities[f'tau_{parameter_letter}'] = PARAMETERS_DEFAULT[corresponding_states] / PARAMETERS_DEFAULT[altered_parameter] / probabilities.index.get_level_values('fold_change')
    probabilities['interval_in_tau_r'] = np.round(
            probabilities.index.get_level_values('interval') 
            * (probabilities.index.get_level_values('fold_change') if altered_parameter == 'r_forward_rate' else 1.)  
            * PARAMETERS_DEFAULT['r_forward_rate'] / PARAMETERS_DEFAULT['r_subcompartments_count']
        ).astype(int)
    probabilities = probabilities.reset_index('interval').set_index(['interval_in_tau_r', f'tau_{parameter_letter}'], append=True)
    probabilities['total spawning'] =  probabilities['<= 6 front spawning'] + probabilities['> 6 front spawning']

    measured_expected_fronts = 1 / probabilities[['<= 6 front spawning', '> 6 front spawning', 'total spawning']].div(1 - probabilities['annihilation by backward front'].to_numpy() / 2 - probabilities['immediate failure'].to_numpy(), axis=0) - 1

    channel_width_to_color = {
        channel_width: f"C{i+4}"
        for i, channel_width in enumerate(measured_expected_fronts.index.unique(level='channel_width'))
    }

    for interval_in_tau_r, data in measured_expected_fronts.reset_index().groupby('interval_in_tau_r'):
        for channel_width, data_part in data.groupby('channel_width'):
            
            predicted_expected_fronts = 1 / (coefs['a_spawning'] * channel_width + coefs['b_spawning']) / channel_length

            ax.plot(
                data_part[f'tau_{parameter_letter}'],
                data_part['total spawning'],
                label=f'{channel_width}',
                color=channel_width_to_color[channel_width],
                marker='s',
                ms=3,
                ls='none',
            )
            if channel_width == 6:
                ax.plot(
                    data_part[data_part['fold_change'] == 1.][f'tau_{parameter_letter}'],
                    data_part[data_part['fold_change'] == 1.]['total spawning'],
                    color=channel_width_to_color[channel_width],
                    marker='o',
                    fillstyle='none',
                    ms=7,
                    ls='none',
                )
            ax.plot(
                predicted_expected_fronts.index,
                predicted_expected_fronts,
                color=channel_width_to_color[channel_width],
                # marker='o',
                alpha=.3,
                ms=3,
                ls='-',
            )
        ax.set_yscale('log')
        ax.set_ylabel('')
        ax.set_title(interval_in_tau_r)
        ax.set_xlabel(f'$\\tau_{parameter_letter.upper()}$ [min]')
        ax.legend(title='channel width $W$')

axs[0].set_ylabel(f'expected number of fronts \nbefore the first spawning event')

plt.savefig(panel_dir / 'fig7.png')
plt.savefig(panel_dir / 'fig7.svg')

