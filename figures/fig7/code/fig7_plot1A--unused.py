# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
 

from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize
from itertools import product
from matplotlib.ticker import MultipleLocator

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

altered_parameter = 'r_forward_rate'
corresponding_states = 'r_subcompartments_count'
parameter_letter = altered_parameter[0]

channel_length = 300
fold_changes = [1.]

probabilities = pd.concat([
        pd.read_csv(data_dir / 'r_rate' / f"{fold_change:.3f}" / 'probabilities.csv').set_index(['channel_length', 'channel_width', 'interval'])
    for fold_change in fold_changes
    ], 
    names=['fold_change'],
    keys=fold_changes,
)

probabilities[f'tau_{parameter_letter}'] = PARAMETERS_DEFAULT[corresponding_states] / PARAMETERS_DEFAULT[altered_parameter] / probabilities.index.get_level_values('fold_change')
probabilities[f'interval_in_tau_{parameter_letter}'] = np.round(probabilities.index.get_level_values('interval') * probabilities.index.get_level_values('fold_change') * PARAMETERS_DEFAULT[altered_parameter] / PARAMETERS_DEFAULT[corresponding_states]).astype(int)
probabilities = probabilities.reset_index('interval').set_index([f'interval_in_tau_{parameter_letter}', f'tau_{parameter_letter}'], append=True)
probabilities['total spawning'] =  probabilities['<= 6 front spawning'] + probabilities['> 6 front spawning']

measured_expected_fronts = 1 / probabilities[['<= 6 front spawning', '> 6 front spawning', 'total spawning']].div(1 - probabilities['annihilation by backward front'].to_numpy() / 2 - probabilities['immediate failure'].to_numpy(), axis=0)


coefficients_rates = pd.read_csv(data_dir_rates / 'fit_coefficients.csv').set_index(['altered_parameter', 'fold_change'])

coefficients_rates['parameter_value'] = [PARAMETERS_DEFAULT[altered_parameter] * fold_change for altered_parameter, fold_change in coefficients_rates.index]
coefficients_rates['n_states'] = list(map(get_corresponding_states, coefficients_rates.index.get_level_values('altered_parameter')))
coefficients_rates['tau'] = coefficients_rates['n_states'] / coefficients_rates['parameter_value']
coefficients_rates['average_n_fronts_per_WL'] = 1 / coefficients_rates['a_spawning'] 

predicted_expected_fronts_per_wl = coefficients_rates.loc[(altered_parameter, 1.), 'average_n_fronts_per_WL']
print(predicted_expected_fronts_per_wl)

channel_width_to_color = {
    channel_width: f"C{i+4}"
    for i, channel_width in enumerate(measured_expected_fronts.index.unique(level='channel_width'))
}


fields_titles = [
 ('> 6 front spawning', 'channel blocking'),
 ('total spawning', 'first spawning event'),
]

fig, axs = subplots_from_axsize(len(measured_expected_fronts.index.unique(level='interval_in_tau_r')), len(fields_titles), sharey=True)


for it, (field, title) in enumerate(fields_titles):
    for ax, (interval_in_tau_r, data) in zip(axs[:, it], measured_expected_fronts.reset_index().groupby('interval_in_tau_r')):
        ax.plot(
            data['channel_width'],
            data[field],
            color='purple',
            marker='o',
        )
        ax.plot(
            data['channel_width'],
            predicted_expected_fronts_per_wl / (data['channel_width'] * channel_length),
            color='purple',
            ls=':',
        )
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_yscale('log')
        ax.set_title(interval_in_tau_r)
        ax.set_xlabel(f'channel width $W$')
        ax.set_ylabel(f'expected number of front till {title}')
        # ax.legend(title='channel width:', title_fontproperties={'weight': 'bold'})

plt.savefig(panel_dir / 'fig7A1.png')
plt.savefig(panel_dir / 'fig7A1.svg')

