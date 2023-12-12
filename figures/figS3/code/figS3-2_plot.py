from pathlib import Path
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize
from matplotlib.ticker import MultipleLocator

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3-2' /'approach1'
out_dir = Path(__file__).parent.parent / 'panels'
out_dir.mkdir(exist_ok=True, parents=True)

spawned_front_fates = pd.read_csv(data_dir / 'spawned_front_fates.csv').set_index(['channel_width', 'channel_length', 'interval', 'simulation_id'])

fig, axs = subplots_from_axsize(1, 2, (4,3), left=.8)

spawned_front_fates.diff(axis=1)[['0', '1', '2','3']].plot.hist(bins=range(0,500,10), alpha=0.3, ax=axs[0])

axs[0].set_xlabel('delay [min]')
axs[0].xaxis.set_minor_locator(MultipleLocator(25))
axs[0].grid(which='both', ls=':')
axs[0].legend(handles=axs[0].get_legend().legend_handles[1:], title='spawned front id')

median_arrival_times = spawned_front_fates.diff(axis=1).groupby(['channel_width', 'channel_length', 'interval']).median()#.reset_index()
counts = spawned_front_fates.diff(axis=1).notna().groupby(['channel_width', 'channel_length', 'interval']).sum()#.reset_index()
median_arrival_times[counts >= 6].reset_index('interval').plot(x='interval', y=['0', '1', '2', '3'], marker='o', alpha=0.3, ax=axs[1])
print(median_arrival_times)

# axs[1].plot(mean_arrival_times['interval'], mean_arrival_times['0'], '-o', label='first pulse')
# axs[1].plot(mean_arrival_times['interval'], mean_arrival_times['1'], '-o', label='second pulse')

axs[1].set_xlim(left=0)
axs[1].set_xlabel('interval [min]')
axs[1].set_ylim(bottom=0, top=250)
axs[1].set_ylabel('delay [min]')
axs[1].grid(which='both', ls=':')
axs[1].legend(loc='upper left')
axs[1].legend(handles=axs[1].get_legend().legend_handles[1:], title='spawned front id')

plt.savefig(out_dir / 'figS3-2.png')
plt.savefig(out_dir / 'figS3-2.svg')


