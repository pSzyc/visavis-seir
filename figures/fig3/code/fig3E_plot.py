import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from subplots_from_axsize import subplots_from_axsize
from cycler import cycler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
sys.path.insert(0, str(Path(__file__).parent)) # in order to be able to import from scripts.py

from scripts.style import *

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3D' / 'approach1'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)


channel_lengths = [300]

fig, axs = subplots_from_axsize(2, 3, (1,.8), top=.4, wspace=.1, hspace=.14)
axs = axs.flatten()

arrival_times = pd.read_csv(data_dir / (f"arrival_times--l-{'-'.join(map(str,channel_lengths))}.csv")).set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'end', 'pulse_id', 'n_simulations', 'n_pulses'])
for ax, (interval, arr_times) in zip(axs[::-1], arrival_times.groupby('interval')['seconds']):
    arr_times.groupby('simulation_id').diff().plot.hist(ax=ax, bins=np.arange(0,600,10), density=True)
    # ylim = ax.get_ylim()
    ylim = (0,0.02)
    ax.vlines(interval * np.arange(600 // interval + 1), *ylim,  color='k', lw=1, ls='--', alpha=.3)
    ax.set_ylim(ylim)
    ax.set_title(f"interval {interval} min", pad=-240, fontweight='bold')
    # ax.legend(title=f"initial interval {interval} min",  loc='center', fontweight='bold')
    # ax.legend([], title=f"initial interval {interval} min",  loc='upper right')
    ax.set_yticks([0, .01, .02])
    ax.set_ylabel('frequency [min$^{-1}$]')
    ax.set_xlabel('interval \nat channel end [min]')



for i in (1,2,4,5):
    axs[i].set_yticklabels([])
    axs[i].set_ylabel('')

for i in (0,1,2):
    axs[i].set_xticklabels([])
    axs[i].set_xlabel('')


plt.savefig(panels_dir / 'fig3E.png')
plt.savefig(panels_dir / 'fig3E.svg')




