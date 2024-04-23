import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from subplots_from_axsize import subplots_from_axsize
from matplotlib.ticker import MultipleLocator

from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.style import *
from scripts.binary import plot_scan
from scripts.handler_tuple_vertical import HandlerTupleVertical


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4EF' / 'approach7'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)

fields = 'c'
k_neighbors = 25
reconstruction = False
suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"
print(f"Drawing plots for suffix: {suffix}")

entropies = pd.read_csv(data_dir /  f'entropies{suffix}.csv').sort_values(['channel_length', 'channel_width', 'interval'])
result = pd.read_csv(data_dir / f'optimized_bitrate{suffix}.csv')
result['max_bitrate_per_hour'] = 60 * result['max_bitrate']

fig, ax = subplots_from_axsize(1, 1, (10, 10))

fig, ax = plot_scan(
    entropies, 
    c_field=['channel_length', 'channel_width'],
    x_field='interval',
    y_field='bitrate_per_hour',
    ms=2,
    ax=ax,
)

plt.scatter(result['optimal_interval'], result['max_bitrate_per_hour'], color='k', marker='o', s=20)

plt.ylim(0,1)

# old:
plt.savefig(panels_dir / f'fig4EF-{suffix}--aux1.svg')
plt.savefig(panels_dir / f'fig4EF-{suffix}--aux1.png')


fig, axs = subplots_from_axsize(1, 2, (1.68, 1.5), left=0.5, wspace=0.5, right=0.01)

ls = np.linspace(result['channel_width'].min(), result['channel_width'].max())

for ch_l_it, (channel_length, result_group) in enumerate(result.groupby('channel_length')):
    result_group.plot(
        'channel_width', 
        'max_bitrate_per_hour', 
        marker='o', 
        color=channel_length_to_color[channel_length], #yerr='max_bitrate_err', capsize=4, 
        ls='-', lw=1, ms=5, ax=axs[0], label=f"{channel_length:.0f}")

axs[0].set_ylabel('max bitrate [bit/h]')
axs[0].set_xlabel('channel width $W$')
axs[0].set_xlim(left=2, right=13)
axs[0].set_ylim(bottom=0, top=1)
axs[0].xaxis.set_major_locator(MultipleLocator(5))
axs[0].legend(title='channel length $L$')
# axs[0].get_legend().set(visible=False)
axs[0].grid(ls=':')


for ch_l_it, (channel_length, result_group) in enumerate(result.groupby('channel_length')):
    result_group[result_group['optimal_interval'] < 300].plot(
        'channel_width', 
        'optimal_interval', 
        marker='o', 
        color=channel_length_to_color[channel_length], 
        #yerr='optimal_interval_err', capsize=4, 
        ls='-', lw=1, ms=5, ax=axs[1], label=f"{channel_length:.0f}")

axs[1].set_ylabel('optimal interval [min]')
axs[1].set_xlabel('channel width $W$')
axs[1].set_xlim(left=2, right=13)
axs[1].set_ylim(bottom=0, top=300)
axs[1].xaxis.set_major_locator(MultipleLocator(5))
# axs[1].legend(title='channel length $L$')
# axs[1].get_legend().set(visible=False)

axs[1].legend()
handles = axs[1].get_legend().legend_handles
axs[1].legend(
    handles=[tuple(handles[i] for i in range(0,3))],
    labels=['colors as \nin panel E'],
    loc='lower right',
    handler_map={tuple: HandlerTupleVertical(ncols=3, vpad=-2.3)},
    # title='colors as in panel A'
    )
axs[1].grid(ls=':')

plt.savefig(panels_dir / f'fig4EF.svg')
plt.savefig(panels_dir / f'fig4EF.png')
# old:
# plt.savefig(panels_dir / f'fig4EF-{suffix}.svg')
# plt.savefig(panels_dir / f'fig4EF-{suffix}.png')

# plt.show()


