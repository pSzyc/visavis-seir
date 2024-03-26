import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from subplots_from_axsize import subplots_from_axsize
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import Polynomial
from matplotlib.ticker import MultipleLocator

from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.style import *
from scripts.binary import plot_scan


channel_length_colors = {
    30: 0,
    100: 1,
    300: 2,
    1000: 3,
}

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4EF' / 'approach5'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)
for fields in 'c', :
    for k_neighbors in (25,):
    #for k_neighbors in (12,18,25,):
        for reconstruction in (False,):
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
                result_group.plot.line('channel_width', 'max_bitrate_per_hour', marker='o', color=f'C{channel_length_colors[channel_length]}', #yerr='max_bitrate_err', capsize=4, 
                 ls='-', lw=1, ax=axs[0], label=f"{channel_length:.0f}")

            axs[0].set_ylabel('max bitrate [bit/h]')
            axs[0].set_xlabel('channel width')
            axs[0].set_xlim(left=2, right=13)
            axs[0].set_ylim(bottom=0, top=1)
            axs[0].xaxis.set_major_locator(MultipleLocator(5))
            axs[0].legend(title='channel length')
            # axs[0].get_legend().set(visible=False)
            axs[0].grid(ls=':')
            

            for ch_l_it, (channel_length, result_group) in enumerate(result.groupby('channel_length')):
                result_group[result_group['optimal_interval'] < 300].plot.line('channel_width', 'optimal_interval', marker='o', color=f'C{channel_length_colors[channel_length]}', #yerr='optimal_interval_err', capsize=4, 
                 ls='-', lw=1, ax=axs[1], label=f"{channel_length:.0f}")
  
            axs[1].set_ylabel('optimal interval [min]')
            axs[1].set_xlabel('channel width')
            axs[1].set_xlim(left=2, right=13)
            axs[1].set_ylim(bottom=0, top=300)
            axs[1].xaxis.set_major_locator(MultipleLocator(5))
            # axs[1].legend(title='channel length')
            axs[1].get_legend().set(visible=False)
            axs[1].grid(ls=':')

            plt.savefig(panels_dir / f'fig4EF.svg')
            plt.savefig(panels_dir / f'fig4EF.png')
            # old:
            # plt.savefig(panels_dir / f'fig4EF-{suffix}.svg')
            # plt.savefig(panels_dir / f'fig4EF-{suffix}.png')

# plt.show()

