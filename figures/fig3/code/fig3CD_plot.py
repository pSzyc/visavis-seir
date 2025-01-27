# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
 

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


fig, ax = subplots_from_axsize(1, 1, (1.8,1.5), left=.7) #1.3

# ----  Fig3C -----

channel_lengths = [30, 100, 300, 1000]

data = pd.read_csv(fig3C_data_dir / (f"n_fronts_received--l-{'-'.join(map(str,channel_lengths))}.csv"))
data['interval_at_channel_end'] = (data['n_pulses'] * data['n_simulations'] * data['interval']) / data['n_fronts_received']

ax.plot((0,300), (0,300), color='k', alpha=.2, ls='-')
for channel_length, data_part in data.groupby('channel_length'):
    ax.plot(data_part['interval'], data_part['interval_at_channel_end'], 
    'o-', label=f"{channel_length}", ms=3,
    )
    t_saturation_loc, t_saturation = data_part.set_index('interval')['interval_at_channel_end'].idxmin(), data_part['interval_at_channel_end'].min()
    ax.plot([0, t_saturation_loc], [t_saturation, t_saturation], ls=':', alpha=0.3, color=channel_length_to_color[channel_length])

t_saturations = data.groupby('channel_length')['interval_at_channel_end'].min()
t_saturations.to_csv(fig3C_data_dir / 't_saturations.csv')

# ax.plot((0,), (0,), color='g', alpha=.2, ls=':', label='final=initial') # just for label

ax.set_ylim(0,350)
ax.set_xlim(0,330)

ax.set_ylabel('average interval\nat channel end [min]')
ax.set_xlabel('initial interval between fronts [min]')
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.legend(title='channel length', loc='lower right')

plt.savefig(panels_dir / 'fig3C.png')
plt.savefig(panels_dir / 'fig3C.svg')


# ----  Fig3D -----

fig, ax = subplots_from_axsize(1, 1, (1.8,1.5), left=.7) #1.3

def cmap_cycler(cmap):
    return cycler('color', plt.get_cmap(cmap)(np.arange(8)/8))


channel_lengths = [1000]
take_intervals = list(range(30,181,30))

data = pd.read_csv(fig3D_data_dir / (f"n_fronts_received--l-{'-'.join(map(str,channel_lengths))}.csv"))

ax.set_prop_cycle(cmap_cycler('Dark2_r'))

for interval, data_part in data.groupby('interval'):
    if interval not in take_intervals: continue
    plt.set_cmap('prism')
    ax.plot(data_part['end'],  data_part['n_fronts_received'] / (data_part['n_pulses'] * data_part['n_simulations']), 'o-', label=f"{interval}", ms=3)


ax.set_ylabel('fraction of transmitted fronts')
ax.set_xlabel('distance along channel [cell layer]')
# ax.set_ylabel('average interval \nat channel end [min]')
ax.set_ylim(0,1.05)
# ax.set_ylim(0,230)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.legend()
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(reversed(handles), reversed(labels), title='initial interval\nbetween pulses', loc='lower right')
leg._legend_box.align = "center"
# ax.set_yscale('log')


plt.savefig(panels_dir / 'fig3D.png')
plt.savefig(panels_dir / 'fig3D.svg')


fig, ax = subplots_from_axsize(1, 1, (1.8,1.3), left=.7)

channel_lengths = [1000]
take_intervals = [30]

data = pd.read_csv(fig3D_data_dir / (f"n_fronts_received--l-{'-'.join(map(str,channel_lengths))}.csv"))

data = data[data['end'] > 30]

ax.set_prop_cycle(cmap_cycler('Dark2_r'))

for interval, data_part in data.groupby('interval'):
    if interval not in take_intervals: continue
    plt.set_cmap('prism')
    ax.plot(np.sqrt(data_part['end']),  1/ (data_part['n_fronts_received'] / (data_part['n_pulses'] * data_part['n_simulations'] *  data_part['interval'])), 'o-', label=f"{interval}", ms=3)


ends = data_part['end'].drop_duplicates().to_numpy()
ax.plot(np.sqrt(ends), np.sqrt(4*ends * 1.68) + 96.5)


ax.set_ylabel('fraction of transmitted fronts')
ax.set_xlabel('$\\sqrt{L}$ [cell layer]')
# ax.set_ylabel('average interval \nat channel end [min]')
# ax.set_ylim(0,1.05)
# ax.set_ylim(0,230)
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.legend()
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(reversed(handles), reversed(labels), title='initial interval\nbetween fronts', loc='lower right')
leg._legend_box.align = "center"
# ax.set_yscale('log')

plt.savefig(panels_dir / 'fig3F.png')
plt.savefig(panels_dir / 'fig3F.svg')


