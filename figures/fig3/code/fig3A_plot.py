# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from pathlib import Path
from shutil import rmtree
from itertools import product

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
sys.path.insert(0, str(Path(__file__).parent)) # in order to be able to import from scripts.py

from scripts.plot_result import plot_result_from_activity
from subplots_from_axsize import subplots_from_axsize
from scripts.style import *

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3A'
out_dir = Path(__file__).parent.parent / 'panels'

channel_widths = [6]
channel_lengths = [300]
intervals = [180, 120, 60]
duration = 5

data_sets = list(product(channel_widths, channel_lengths, intervals))


fig, axs = subplots_from_axsize(1, len(data_sets), (1.8,3.6), wspace=0.15, left=0.6, right=0.1, top=0.3)

for it, (channel_width, channel_length, interval) in enumerate(data_sets):
    activity = pd.read_csv(data_dir / f'fig3A_w-{channel_width}-l-{channel_length}-interval-{interval}--activity.csv').set_index(['frame', 'seconds'])
    plot_result_from_activity(activity, ax=axs[it], cmap='Greys', transpose=True, show=False)

    experiment_time = activity.index.get_level_values('seconds').max() + 1
    axs[it].set_axis_on()
    axs[it].set_yticks(list(range(0,experiment_time // duration, 180)))
    axs[it].yaxis.set_minor_locator(MultipleLocator(interval // duration))
    if it == 0:
        axs[it].set_ylabel('time [min]')
        axs[it].set_yticklabels(list(range(0,experiment_time, duration * 180)))
    else:
        axs[it].set_ylabel('')
        axs[it].set_yticklabels([])
    axs[it].set_title(f"interval {interval} min", loc='left', pad=-20, fontweight='bold')
    axs[it].set_xticks([0,channel_length / 2, channel_length])
    axs[it].set_xlabel('position along channel')

plt.savefig(out_dir / 'fig3A.png')
plt.savefig(out_dir / 'fig3A.svg')
            
