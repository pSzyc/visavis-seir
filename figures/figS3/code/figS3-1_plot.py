from pathlib import Path
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize
from matplotlib.ticker import MultipleLocator

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3' / 'figS3-1' / 'approach1'
out_dir = Path(__file__).parent.parent / 'panels'
out_dir.mkdir(exist_ok=True, parents=True)

# MEAN

mean_arrival_times = pd.read_csv(data_dir / 'mean_arrival_times.csv')
both_arrived_pulse_fates = pd.read_csv(data_dir / 'both_arrived_pulse_fates.csv').set_index(['channel_width', 'channel_length', 'interval', 'simulation_id'])

fig, axs = subplots_from_axsize(1, 2, (4,3), left=.8)

for it, (interval, data) in enumerate(both_arrived_pulse_fates.groupby('interval')):
    data.diff(axis=1).plot.hist(bins=range(0,500,10), bottom=it*1000, ax=axs[0])

axs[0].set_xlabel('delay [min]')
axs[0].xaxis.set_minor_locator(MultipleLocator(25))
axs[0].grid(which='both', ls=':')
axs[0].get_legend().set_visible(False)

ax=axs[1]

ax.plot(mean_arrival_times['interval'], mean_arrival_times['0'], '-o', label='first pulse')
ax.plot(mean_arrival_times['interval'], mean_arrival_times['1'], '-o', label='second pulse')
ax.plot(mean_arrival_times['interval'], mean_arrival_times['0'] + mean_arrival_times['interval'], ':', label='first pulse + interval')

ax.set_xlim(left=0)
ax.set_xlabel('interval [min]')
ax.set_ylabel('arrival time [min]')
ax.legend(loc='upper left')

eps=10
# ax.arrow(75,1080+eps,0,145-2*eps, width=0.001, color="k", 
#             head_width=3, head_length=10, length_includes_head=True,
#             arrowstyle="<->")
ax.annotate("", xy=(75, 1225), xytext=(75, 1080),
            arrowprops=dict(arrowstyle="<->"))
ax.annotate("145 min", xy=(75, 1152.5), xytext=(85, 1152.5))
plt.savefig(out_dir / 'figS3.png')
plt.savefig(out_dir / 'figS3.svg')

print(mean_arrival_times.set_index('interval').pipe(lambda df: df['1'] - df['0']))
print(mean_arrival_times.set_index('interval').pipe(lambda df: df['1'] - df['0']))


# MEDIAN


median_arrival_times = pd.read_csv(data_dir / 'median_arrival_times.csv')

fig, ax = subplots_from_axsize(1, 1, (4,3), left=.8)

ax.plot(median_arrival_times['interval'], median_arrival_times['0'], '-o', label='first pulse')
ax.plot(median_arrival_times['interval'], median_arrival_times['1'], '-o', label='second pulse')
ax.plot(median_arrival_times['interval'], median_arrival_times['0'] + median_arrival_times['interval'], ':', label='first pulse + interval')

ax.set_xlim(left=0)
ax.set_xlabel('interval [min]')
ax.set_ylabel('arrival time [min]')
ax.legend(loc='upper left')

eps=10
# ax.arrow(75,1080+eps,0,145-2*eps, width=0.001, color="k", 
#             head_width=3, head_length=10, length_includes_head=True,
#             arrowstyle="<->")
ax.annotate("", xy=(75, 1225), xytext=(75, 1080),
            arrowprops=dict(arrowstyle="<->"))
ax.annotate("145 min", xy=(75, 1152.5), xytext=(85, 1152.5))
plt.savefig(out_dir / 'figS3--median.png')
plt.savefig(out_dir / 'figS3--median.svg')

