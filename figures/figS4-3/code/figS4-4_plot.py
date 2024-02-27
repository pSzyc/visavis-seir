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


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS4-3' / 'figS4-3' / 'approach2'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)
differences = pd.read_csv(data_dir / 'difference_trajectories.csv').set_index(['channel_width', 'channel_length', 'interval'])

fig, ax = subplots_from_axsize(1, 1, (4, 3), left=0.5, wspace=0.5)
differences_grouped = differences.reindex(columns=map(str,range(5,1500,5))).stack().unstack(['channel_width', 'channel_length', 'interval'])*3.6
differences_grouped.index = np.sqrt(np.array(list(map(int,differences_grouped.index))))
differences_grouped.plot(ms=3, ax=ax)
ax.get_legend().set(visible=False)
for (_, _,interval), first_val in differences_grouped.iloc[0].items():
    ax.annotate(interval, (0, first_val), horizontalalignment='right', verticalalignment='center_baseline')


# ax.xaxis.set_minor_locator(MultipleLocator(20))
plt.grid(which='both', ls=':')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"$\\sqrt{{{x**2:.0f}}}$"))
ax.set_xlabel('${\\sqrt{\\mathrm{simulation~time}}}$ [min$^\\frac{1}{2}$]')
ax.set_ylabel('average distance between fronts [min]')


plt.savefig(panels_dir / f'figS4-4.svg')
plt.savefig(panels_dir / f'figS4-4.png')

fig, ax = subplots_from_axsize(1, 1, (8, 6), left=0.5, wspace=0.5)

for (channel_width, channel_length, interval), data in differences.reindex(columns=map(str,range(20,2800,5))).stack().groupby(['channel_width', 'channel_length', 'interval']):
    ax.plot(data * 3.6, data.diff().rolling(5, center=True).mean() / 5 *3.6)


plt.savefig(panels_dir / f'figS4-4--aux1.svg')
plt.savefig(panels_dir / f'figS4-4--aux1.png')


