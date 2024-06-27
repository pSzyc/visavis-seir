# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from subplots_from_axsize import subplots_from_axsize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.handler_tuple_vertical import HandlerTupleVertical

plt.rcParams["font.sans-serif"] = ['Carlito']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Carlito'
plt.rcParams['font.size'] = 8


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3' / 'figS3' / 'approach11'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)
differences = pd.read_csv(data_dir / 'difference_in_reaching_times.csv').set_index(['channel_width', 'channel_length', 'interval']).assign(sqrt_h=lambda df: np.sqrt(df['h']))

fig, axs = subplots_from_axsize(1, 2, (2.7, 2), left=0.5, wspace=0.7)

ax = axs[0]

differences_filtered = differences[differences['h'].gt(2) & differences['count'].gt(30) & (differences.index.get_level_values('interval') >= 45)]
differences_filtered.groupby(['channel_width', 'channel_length', 'interval']).plot('sqrt_h', 'mean', ax=ax)

differences_filtered = differences[differences['h'].gt(0) & differences['count'].gt(30) & (differences.index.get_level_values('interval') >= 45)]
differences_filtered.groupby(['channel_width', 'channel_length', 'interval']).apply(lambda x: ax.plot(x['sqrt_h'], x['min'].rolling(5, center=True).mean(), alpha=.3))#.plot('sqrt_h', 'min', ax=ax, alpha=.3)

n_series = len(differences_filtered.index.unique())
handles = ax.get_lines()#ax.get_legend().legend_handles
ax.legend(
    handles=[tuple(handles[:n_series]), tuple(handles[n_series:2*n_series])], #title='channel length $L$',
    labels=['average', 'minimum'],
    handler_map={tuple: HandlerTupleVertical(nrows=1, vpad=-2.3)},
    title='interval between fronts'
    )

ax.grid(which='both', ls=':')
ax.xaxis.set_ticks(np.sqrt(np.array([1,10,30,100,300,1000])))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"$\\sqrt{{{x**2:.0f}}}$"))
ax.set_xlabel('${\\sqrt{\\mathrm{distance~along~channel}}}$')
ax.set_ylabel('interval between fronts [min]')

reaching_time_by_h = pd.read_csv(data_dir / 'reaching_time_by_h.csv')
time_of_reaching_300 = reaching_time_by_h[reaching_time_by_h['h'] == 30]
time_of_reaching_300.plot('interval', ['first_pulse_reaching_time', 'second_pulse_reaching_time'], marker='o', ax=axs[1])
axs[1].plot(time_of_reaching_300['interval'], time_of_reaching_300['first_pulse_reaching_time'] + time_of_reaching_300['interval'], ls=':')

axs[1].annotate("", xy=(65, 205), xytext=(65, 105),
            arrowprops=dict(arrowstyle="<->"))
axs[1].annotate("100 min", xy=(65, 152.5), xytext=(70, 152.5))


axs[1].legend(['first front', 'second front', 'first front + initial interval'])
axs[1].set_xlabel('initial interval [min]')
axs[1].set_ylabel('average interval at $L = 300$ [min]')

plt.savefig(panels_dir / f'figS3.svg')
plt.savefig(panels_dir / f'figS3.png')

fig, ax = subplots_from_axsize(1, 1, (8, 6), left=0.5, wspace=0.5)

for (channel_width, channel_length, interval), data in differences_filtered.groupby(['channel_width', 'channel_length', 'interval']):
    ax.plot(data['mean'], data['mean'].diff().rolling(15, center=True).mean() / 5)

ax.set_yscale('log')

plt.savefig(panels_dir / f'figS3--aux1.svg')
plt.savefig(panels_dir / f'figS3--aux1.png')


