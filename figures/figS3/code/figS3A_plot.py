import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from subplots_from_axsize import subplots_from_axsize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter

from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

plt.rcParams["font.sans-serif"] = ['Carlito']
# plt.rcParams['mathtext.fontset'] = 'Carlito'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Carlito'
plt.rcParams['font.size'] = 8


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3' / 'figS3A' / 'approach1'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)
# differences = pd.read_csv(data_dir / 'difference_trajectories.csv').set_index(['channel_width', 'channel_length', 'interval'])
differences = pd.read_csv(data_dir / 'difference_in_reaching_times.csv').set_index(['channel_width', 'channel_length', 'interval']).assign(sqrt_h=lambda df: np.sqrt(df['h']))

# reaching_times = pd.read_csv(data_dir / 'reaching_times.csv')#.set_index(['channel_width', 'channel_length', 'interval', 'simulation_id'])

fig, axs = subplots_from_axsize(1, 2, (2.7, 2), left=0.5, wspace=0.7)

ax = axs[0]

differences_filtered = differences[differences['h'].gt(2) & differences['count'].gt(30) & (differences.index.get_level_values('interval') >= 45)]
differences_filtered.groupby(['channel_width', 'channel_length', 'interval']).plot('sqrt_h', 'mean', ax=ax)
differences_filtered = differences[differences['h'].gt(0) & differences['count'].gt(30) & (differences.index.get_level_values('interval') >= 45)]
differences_filtered.groupby(['channel_width', 'channel_length', 'interval']).apply(lambda x: ax.plot(x['sqrt_h'], x['min'].rolling(5, center=True).mean(), alpha=.3))#.plot('sqrt_h', 'min', ax=ax, alpha=.3)
ax.get_legend().set(visible=False)

ax.grid(which='both', ls=':')
ax.xaxis.set_ticks(np.sqrt(np.array([1,10,30,100,300,1000])))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"$\\sqrt{{{x**2:.0f}}}$"))
ax.set_xlabel('${\\sqrt{\\mathrm{distance~along~channel}}}$')
ax.set_ylabel('interval between fronts [min]')
# ax.set_xscale('log')

# reaching_time_by_h = reaching_times.groupby(['channel_width', 'channel_length', 'interval', 'h']).mean().reset_index()
# reaching_time_by_h.to_csv(data_dir / 'reaching_time_by_h.csv')
reaching_time_by_h = pd.read_csv(data_dir / 'reaching_time_by_h.csv')
time_of_reaching_300 = reaching_time_by_h[reaching_time_by_h['h'] == 30]
time_of_reaching_300.plot('interval', ['first_pulse_reaching_time', 'second_pulse_reaching_time'], marker='o', ax=axs[1])
axs[1].plot(time_of_reaching_300['interval'], time_of_reaching_300['first_pulse_reaching_time'] + time_of_reaching_300['interval'], ls=':')

# axs[1].annotate("", xy=(75, 1230), xytext=(75, 1085),
#             arrowprops=dict(arrowstyle="<->"))
# axs[1].annotate("145 min", xy=(75, 1152.5), xytext=(85, 1152.5))

axs[1].annotate("", xy=(65, 205), xytext=(65, 105),
            arrowprops=dict(arrowstyle="<->"))
axs[1].annotate("100 min", xy=(65, 152.5), xytext=(70, 152.5))


axs[1].legend(['first pulse', 'second pules', 'first pulse + initial interval'])
axs[1].set_xlabel('initial interval [min]')
axs[1].set_ylabel('mean interval at $L = 300$ [min]')

plt.savefig(panels_dir / f'figS3A.svg')
plt.savefig(panels_dir / f'figS3A.png')

fig, ax = subplots_from_axsize(1, 1, (8, 6), left=0.5, wspace=0.5)

for (channel_width, channel_length, interval), data in differences_filtered.groupby(['channel_width', 'channel_length', 'interval']):
    ax.plot(data['mean'], data['mean'].diff().rolling(15, center=True).mean() / 5)

ax.set_yscale('log')

plt.savefig(panels_dir / f'figS3A--aux1.svg')
plt.savefig(panels_dir / f'figS3A--aux1.png')


