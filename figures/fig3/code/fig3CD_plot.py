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

fig3C_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3C' / 'approach1'
fig3D_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3D' / 'approach1'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)


fig, axs = subplots_from_axsize(2, 1, (1.8,1.3), left=.7)

# ----  Fig3C -----

channel_lengths = [30, 100, 300, 1000]

data = pd.read_csv(fig3C_data_dir / (f"n_fronts_received--l-{'-'.join(map(str,channel_lengths))}.csv"))

for channel_length, data_part in data.groupby('channel_length'):
    axs[0].plot(data_part['interval'],  1 / data_part['n_fronts_received'] * (data_part['n_pulses'] * data_part['n_simulations'] * data_part['interval']), 
    'o-', label=f"{channel_length}", ms=3,
    )

axs[0].set_ylim(0,350)

axs[0].set_ylabel('average interval\nat channel end [min]')
axs[0].set_xlabel('initial interval between pulses [min]')
# axs[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
axs[0].legend(title='channel length', loc='lower right')


# ----  Fig3D -----


def cmap_cycler(cmap):
    return cycler('color', plt.get_cmap(cmap)(np.arange(8)/8))


channel_lengths = [1000]
take_intervals = list(range(30,181,30))

data = pd.read_csv(fig3D_data_dir / (f"n_fronts_received--l-{'-'.join(map(str,channel_lengths))}.csv"))

axs[1].set_prop_cycle(cmap_cycler('Dark2_r'))

for interval, data_part in data.groupby('interval'):
    if interval not in take_intervals: continue
    plt.set_cmap('prism')
    axs[1].plot(data_part['end'],  data_part['n_fronts_received'] / (data_part['n_pulses'] * data_part['n_simulations']), 'o-', label=f"{interval}", ms=3)


axs[1].set_ylabel('fraction of fronts')
axs[1].set_xlabel('distance along channel [cell layers]')
# axs[1].set_ylabel('average interval \nat channel end [min]')
axs[1].set_ylim(0,1.05)
# axs[1].set_ylim(0,230)
axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
axs[1].legend()
handles, labels = axs[1].get_legend_handles_labels()
leg = axs[1].legend(reversed(handles), reversed(labels), title='initial interval\nbetween pulses', loc='lower right')
leg._legend_box.align = "center"
# axs[1].set_yscale('log')


plt.savefig(panels_dir / 'fig3CD.png')
plt.savefig(panels_dir / 'fig3CD.svg')

