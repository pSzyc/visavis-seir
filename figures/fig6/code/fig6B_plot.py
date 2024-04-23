import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from pathlib import Path
from shutil import rmtree
from itertools import product
from subplots_from_axsize import subplots_from_axsize

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
sys.path.insert(0, str(Path(__file__).parent)) # in order to be able to import from scripts.py

from scripts.plot_result import plot_result_from_activity
from scripts.defaults import PARAMETERS_DEFAULT
from scripts.style import *

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig6'/ 'fig6B' / 'approach1'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(exist_ok=True, parents=True)

channel_width = 6
channel_length = 300
# intervals = [100, 150, 200]
duration = 5

data_sets = np.array([[
    (150, {'r_forward_rate': 4/tau_r})
    for tau_r in [60,40,30]
 ], [
    (150, {'e_forward_rate': 4/tau_e})
    for tau_e in [4,6,10]
 ]])


def parameter_to_label(param_with_value):
    param, value = param_with_value

    molecule = param[0]

    if param == 'c_rate':
        param_label = f"$\\tau_{{{act}}}$"
        presented_value = 1 / value
        return f"{param_label} = {presented_value:.0f}"


    if param.endswith('_forward_rate'):
        param_label = f"$\\tau_{molecule.upper()}$"
        presented_value = PARAMETERS_DEFAULT[f"{molecule}_subcompartments_count"] / value
        return f"{param_label} = {presented_value:.0f}"

    if param.endswith('_subcompartments_count'):
        param_label = f"$n_{molecule.upper()}$"
        presented_value = value
        return f"{param_label} = {presented_value:.0f}"




fig, axs = subplots_from_axsize(*(data_sets.shape[:-1]), (1.8,2.1), wspace=0.15, hspace=.2, left=0.6, right=0.1, top=0.3)

for ax, (interval, parameters_update) in zip(axs.flatten(), data_sets.reshape(-1, data_sets.shape[-1])):
    activity = pd.read_csv(data_dir / '/'.join(map(lambda param_upd: f"{param_upd[0]}/{param_upd[1]:.3f}", sorted(parameters_update.items()))) /'activity.csv').set_index(['frame', 'seconds'])
    plot_result_from_activity(activity, ax=ax, cmap='Greys', transpose=True, show=False)

    title = ', '.join(map(parameter_to_label, sorted(parameters_update.items())))

    experiment_time = activity.index.get_level_values('seconds').max() + 1
    ax.set_axis_on()
    ax.set_yticks(list(range(0,experiment_time // duration, 180)))
    ax.yaxis.set_minor_locator(MultipleLocator(interval // duration))
    ax.set_ylabel('')
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_xticks([0,channel_length / 2, channel_length])

    # ax.set_ylabel(f'{interval = } min')
    ax.set_title(title, loc='left', pad=-20, fontweight='bold')

    # ax2 = ax.twinx()
for ax in axs[-1]:
    ax.set_xlabel('position along channel')
    ax.set_xticklabels(map(int, [0,channel_length / 2, channel_length]))

for ax in axs[:, 0]:
    ax.set_ylabel('time [min]')
    ax.set_yticklabels(list(range(0,experiment_time, duration * 180)))

plt.savefig(panels_dir / 'fig6B.png')
plt.savefig(panels_dir / 'fig6B.svg')
            
