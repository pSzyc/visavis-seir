# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from subplots_from_axsize import subplots_from_axsize
from sklearn.linear_model import LinearRegression
from pathlib import Path
from matplotlib.ticker import MultipleLocator

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.binary import plot_scan
from scripts.entropy_utils import get_cycle_time_std, get_cycle_time


plt.rcParams["font.sans-serif"] = ['Carlito']
plt.rcParams['font.size'] = 8   
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Carlito'

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' /'fig4CD' / 'approach6' 
panels_dir = Path(__file__).parent.parent / 'panels' 
panels_dir.mkdir(parents=True, exist_ok=True)


fields = 'c'
k_neighbors = 25
reconstruction = False
suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"
print(f"Drawing plots for suffix: {suffix}")

entropies = pd.read_csv(data_dir / f'entropies{suffix}.csv').sort_values(['channel_width', 'channel_length', 'interval'])
result = pd.read_csv(data_dir / f'optimized_bitrate{suffix}.csv'
    ).assign(ratio=lambda df: 1/(df['max_bitrate_inv'] / df['optimal_interval'])
    ).assign(max_bitrate_per_hour = lambda df: df['max_bitrate'] * 60
    ).assign(optimal_interval_err = lambda df: df['optimal_interval'] / 15
    )


fig, ax = plot_scan(
    entropies, 
    c_field=['channel_width', 'channel_length'],
    x_field='interval',
    y_field='bitrate_per_hour',
    ms='2',
)
plt.scatter(result['optimal_interval'], 60*result['max_bitrate'], color='k', marker='o', s=20)
plt.ylim(0,1)

# old:
plt.savefig(panels_dir / f'fig4CD{suffix}--aux1.svg')
plt.savefig(panels_dir / f'fig4CD{suffix}--aux1.png')

fit_max_bitrate_sqrt_inv = LinearRegression().fit(result[['channel_length_sqrt']].to_numpy()[:-1], result[['max_bitrate_inv']].to_numpy()[:-1])
predict_max_bitrate_sqrt_inv = fit_max_bitrate_sqrt_inv.predict(result[['channel_length_sqrt']].to_numpy())
print(f"max_bitrate = 1 / (sqrt(l)/{1/fit_max_bitrate_sqrt_inv.coef_[0][0]:.2g} + {fit_max_bitrate_sqrt_inv.intercept_[0]:.2g})")

fit_optimal_interval_sqrt = LinearRegression().fit(result[['channel_length_sqrt']].to_numpy(), result[['optimal_interval']].to_numpy())
predict_optimal_interval_sqrt = fit_optimal_interval_sqrt.predict(result[['channel_length_sqrt']].to_numpy())
print(f"optimal_interval = {fit_optimal_interval_sqrt.coef_[0][0]:.2g}*sqrt(l) + {fit_optimal_interval_sqrt.intercept_[0]:.2g}")


fig, axs = subplots_from_axsize(1, 2, (1.68, 1.5), left=0.5, wspace=0.5, right=0.1)

result.plot('channel_length_sqrt', 'max_bitrate_per_hour',  marker='o', color='maroon', ls='-', ax=axs[0])
# axs[0].plot(result['channel_length_sqrt'], predict_max_bitrate_sqrt_inv, alpha=0.4, color='k')
axs[0].plot(np.sqrt(np.array([1,1000])), [60/(get_cycle_time())]*2, alpha=0.4, color='purple', ls='-')
axs[0].plot(np.sqrt(np.array([1,1000])), [60/(get_cycle_time() + get_cycle_time_std())]*2, alpha=0.4, color='purple', ls='--')
axs[0].annotate('1 bit / $T_{\\mathrm{cycle}}$', (np.sqrt(1000), 60/(get_cycle_time())), xytext=(0,-2), textcoords='offset points', horizontalalignment='right', verticalalignment='top', alpha=0.4)
axs[0].annotate('1 bit / $T_{\\mathrm{R}}$ = 1 bit / ($T_{\\mathrm{cycle}} +\sigma_{\\mathrm{cycle}}$)', (np.sqrt(1000), 60/(get_cycle_time() + get_cycle_time_std())), xytext=(0,-2), textcoords='offset points', horizontalalignment='right', verticalalignment='top', alpha=0.4)
result.plot('channel_length_sqrt', 'optimal_interval', marker='o', color='maroon', ls='-', ax=axs[1])
axs[1].plot(np.sqrt(np.array([1,1000])), [(get_cycle_time())]*2, alpha=0.4, color='purple', ls='-')
axs[1].plot(np.sqrt(np.array([1,1000])), [(get_cycle_time() + get_cycle_time_std())]*2, alpha=0.4, color='purple', ls='--')
axs[1].annotate('$T_{\\mathrm{cycle}}$', (np.sqrt(1000), (get_cycle_time())), xytext=(0,-2), textcoords='offset points', horizontalalignment='right', verticalalignment='top', alpha=0.4)
axs[1].annotate('$T_{\\mathrm{R}} = T_{\\mathrm{cycle}} +\sigma_{\\mathrm{cycle}}$', (np.sqrt(1000), (get_cycle_time() + get_cycle_time_std())), xytext=(0,-2), textcoords='offset points', horizontalalignment='right', verticalalignment='top', alpha=0.4)

axs[0].set_xlabel('$\sqrt{\mathrm{channel~length}~L}$')
axs[1].set_xlabel('$\sqrt{\mathrm{channel~length}~L}$')
axs[0].set_ylabel('max bitrate [bit/h]')
axs[1].set_ylabel('optimal interval $T_{\\mathrm{slot}}$ [min]')
axs[0].legend(labels=['channel width $W$ = 6'])
axs[1].legend(labels=['channel width $W$ = 6'])

axs[0].set_ylim(0,1)
axs[1].set_ylim(0,300)
axs[0].set_xticks(np.sqrt(np.array([1,30,100,300,1000])))
axs[1].set_xticks(np.sqrt(np.array([1,30,100,300,1000])))
axs[0].set_xticklabels(map("$\sqrt{{{:d}}}$".format, [1,30,100,300,1000]))
axs[1].set_xticklabels(map("$\sqrt{{{:d}}}$".format, [1,30,100,300,1000]))
axs[1].yaxis.set_major_locator(MultipleLocator(50))   
axs[0].grid(ls=':')
axs[1].grid(ls=':')


plt.savefig(panels_dir / f'fig4CD.svg')
plt.savefig(panels_dir / f'fig4CD.png')


fig, axs = subplots_from_axsize(1, 2, (2.75, 2))

axs[0].plot(result['channel_length_sqrt'], 1/result['max_bitrate'], marker='o', color='k', ls='none')
axs[0].plot(result['channel_length_sqrt'], predict_max_bitrate_sqrt_inv)
axs[1].plot(result['channel_length_sqrt'], result['optimal_interval'], marker='o', color='k', ls='none')
axs[1].plot(result['channel_length_sqrt'], predict_optimal_interval_sqrt)
axs[0].set_xlabel('1 / sqrt(channel length)')
axs[0].set_ylabel('max_bitrate [bit/min]')
# axs[0].set_xlim(left=0)
# axs[0].set_ylim(bottom=0)



fig, axs = subplots_from_axsize(1, 2, (40 / 25.4, 30 / 25.4))

ls = np.linspace(result['channel_length'].min(), result['channel_length'].max())
result.plot.line('channel_length', 'max_bitrate', marker='o', color='k', ls='none', ax=axs[0])
axs[0].plot(
    ls,
    1 / fit_max_bitrate_sqrt_inv.predict(np.sqrt(ls).reshape(-1,1)),
    label=f"1 / [sqrt(L) / {1/fit_max_bitrate_sqrt_inv.coef_[0][0]:.1g} + {fit_max_bitrate_sqrt_inv.intercept_[0]:.1g}]",
    )

axs[0].set_xlim(left=0)
axs[0].set_ylim(bottom=0, top=1)
axs[0].legend()

result.plot.line('channel_length', 'optimal_interval', marker='o', color='k', ls='none', ax=axs[1])
axs[1].plot(
    ls,
    fit_optimal_interval_sqrt.predict(np.sqrt(ls).reshape(-1,1)),
    label=f"{fit_optimal_interval_sqrt.coef_[0][0]:.1g} × sqrt(L) + {fit_optimal_interval_sqrt.intercept_[0]:.1g}",
    )
axs[1].set_xlim(left=0)
axs[1].set_ylim(bottom=0, top=200)
axs[1].legend()


