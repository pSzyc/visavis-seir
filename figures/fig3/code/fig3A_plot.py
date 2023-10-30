import pandas as pd
from matplotlib import pyplot as plt

from pathlib import Path
from shutil import rmtree
from itertools import product

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
sys.path.insert(0, str(Path(__file__).parent)) # in order to be able to import from scripts.py

from scripts.plot_result import plot_result_from_states
from fig3A_get_data import channel_widths, channel_lengths, intervals, duration
from subplots_from_axsize import subplots_from_axsize

data_dir = Path(__file__).parent.parent / 'data'
out_dir = Path(__file__).parent.parent / 'panels'

data_sets = list(product(channel_widths, channel_lengths, intervals))

fig, axs = subplots_from_axsize(len(data_sets), 1, (3,1.5), hspace=0.15, left=0.4, right=0.8)

for it, (channel_width, channel_length, interval) in enumerate(data_sets):
    states = pd.read_csv(data_dir / f'fig3A_w-{channel_width}-l-{channel_length}-interval-{interval}.csv')
    plot_result_from_states(states, ax=axs[it])

    experiment_time = states['seconds'].max() + 1
    axs[it].set_xlabel('time' if it == len(data_sets)-1 else '')
    axs[it].set_axis_on()
    axs[it].set_xticks(list(range(0,experiment_time // duration, 300)))
    if it == len(data_sets) - 1:
        axs[it].set_xticklabels(list(range(0,experiment_time, duration * 300)))
    else:
        axs[it].set_xticklabels([])
    axs[it].set_yticks([])
    axs[it].set_ylabel(f'{interval = } min')
    ax2 = axs[it].twinx()
    ax2.set_yticks([0,channel_length / 2, channel_length])
    ax2.set_ylabel('position along channel', rotation=-90, labelpad=16.)

plt.savefig(out_dir / 'fig3A.png')
plt.savefig(out_dir / 'fig3A.svg')
            
