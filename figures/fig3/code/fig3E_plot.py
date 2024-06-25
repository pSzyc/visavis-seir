import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from subplots_from_axsize import subplots_from_axsize

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py

from scripts.style import *

fig3C_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3C' / 'approach1'
fig3D_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3D' / 'approach1'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)


channel_length = 300

fig, axs = subplots_from_axsize(2, 3, (1,.7), left=0.44, top=0.12, wspace=.14, hspace=.24)
axs = axs.flatten()

arrival_times = pd.read_csv(fig3D_data_dir / (f"arrival_times--l-{channel_length}.csv")).set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'end', 'pulse_id', 'n_simulations', 'n_pulses'])
t_saturations = pd.read_csv(fig3C_data_dir / 't_saturations.csv').set_index(['channel_length'])['interval_at_channel_end']
for ax, (interval, arr_times) in zip(axs[::-1], arrival_times.groupby('interval')['seconds']):
    arr_times.groupby('simulation_id').diff().plot.hist(ax=ax, bins=np.arange(0,600,10), density=True, color='coral')
    ylim = (0,0.02)
    ax.vlines(interval * np.arange(600 // interval + 1), *ylim,  color='k', lw=1, ls='--', alpha=.3)
    ax.vlines([t_saturations.loc[channel_length]], *ylim,  color='pink', lw=1, ls='-')
    ax.set_ylim(ylim)
    ax.set_title(f"interval {interval} min", pad=-240, fontweight='bold')
    ax.set_yticks([0, .01, .02])
    ax.set_yticklabels([])
    ax.set_ylabel('')
    ax.set_xlabel('interval \nat channel end [min]')



for i in (1,2,4,5):
    axs[i].set_yticklabels([])
    axs[i].set_ylabel('')

for i in (0,1,2):
    axs[i].set_xticklabels([])
    axs[i].set_xlabel('')


plt.savefig(panels_dir / 'fig3E.png')
plt.savefig(panels_dir / 'fig3E.svg')




