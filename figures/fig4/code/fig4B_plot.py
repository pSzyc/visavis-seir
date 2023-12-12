import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from subplots_from_axsize import subplots_from_axsize
from sklearn.linear_model import LinearRegression
from warnings import warn

from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.binary import plot_scan

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' /'fig4B' / 'approach1'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)
# suffix = "-c"
# for fields in 'c', 'rl', 'cm', 'cp', 'cmp':
for fields in 'c', :
    for k_neighbors in (25,):
        # for reconstruction in (True, False):
        for reconstruction in (False,):
            suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"
            print(f"Drawing plots for suffix: {suffix}")

            entropies = pd.read_csv(data_dir / f'entropies{suffix}.csv')
            result = pd.read_csv(data_dir / f'optimized_bitrate{suffix}.csv')

            fig, ax = plot_scan(
                entropies, 
                c_field='channel_length',
                x_field='interval',
                y_field='bitrate_per_hour',
                ms='2',
            )
            plt.scatter(result['optimal_interval'], 60*result['max_bitrate'], color='k', marker='o', s=20)
            plt.ylim(0,1)

            plt.savefig(panels_dir / f'fig4B{suffix}--aux1.svg')
            plt.savefig(panels_dir / f'fig4B{suffix}--aux1.png')



            fit_max_bitrate_sqrt_inv = LinearRegression().fit(result[['channel_length_sqrt']].to_numpy(), result[['max_bitrate_inv']].to_numpy())
            predict_max_bitrate_sqrt_inv = fit_max_bitrate_sqrt_inv.predict(result[['channel_length_sqrt']].to_numpy())
            print(f"max_bitrate = 1 / (sqrt(l)/{1/fit_max_bitrate_sqrt_inv.coef_[0][0]:.2g} + {fit_max_bitrate_sqrt_inv.intercept_[0]:.2g})")

            fit_optimal_interval_sqrt = LinearRegression().fit(result[['channel_length_sqrt']].to_numpy(), result[['optimal_interval']].to_numpy())
            predict_optimal_interval_sqrt = fit_optimal_interval_sqrt.predict(result[['channel_length_sqrt']].to_numpy())
            print(f"optimal_interval = {fit_optimal_interval_sqrt.coef_[0][0]:.2g}*sqrt(l) + {fit_optimal_interval_sqrt.intercept_[0]:.2g}")

            fig, axs = subplots_from_axsize(1, 2, (4,3))

            result.plot.line('channel_length_sqrt', 'max_bitrate_inv', marker='o', color='k', ls='none', ax=axs[0])
            axs[0].plot(result['channel_length_sqrt'], predict_max_bitrate_sqrt_inv, alpha=0.4, color='k')
            result.plot.line('channel_length_sqrt', 'optimal_interval', marker='o', color='k', ls='none', ax=axs[1])
            axs[1].plot(result['channel_length_sqrt'], predict_optimal_interval_sqrt, alpha=0.4, color='k')
            axs[0].set_xlabel('$\sqrt{\mathrm{channel~length}}$')
            axs[0].set_ylabel('(maximal bitrate)$^{-1}$ [min/bit]')
            axs[1].set_xlabel('$\sqrt{\mathrm{channel~length}}$')
            axs[1].set_ylabel('optimal interval [min]')
            axs[0].legend(
                ['simulations', f"${fit_max_bitrate_sqrt_inv.coef_[0][0]:.02g} × \sqrt{{L}} + {fit_max_bitrate_sqrt_inv.intercept_[0]:.02g}$"],
                # ['simulations', f"$\sqrt{{L~/~{1/fit_max_bitrate_sqrt_inv.coef_[0][0]**2:.03g}}} + {fit_max_bitrate_sqrt_inv.intercept_[0]:.03g}$"],
                loc='upper left',
                )
            axs[1].legend(
                ['simulations', f"${fit_optimal_interval_sqrt.coef_[0][0]:.02g} × \sqrt{{L}} + {fit_optimal_interval_sqrt.intercept_[0]:.02g}$"],
                # ['simulations', f"$\sqrt{{L~/~{1/fit_optimal_interval_sqrt.coef_[0][0]**2:.03g}}} + {fit_optimal_interval_sqrt.intercept_[0]:.03g}$"],
                loc='upper left',
                )
            axs[0].set_ylim(0,600)
            axs[1].set_ylim(0,300)
            axs[0].set_xticks(np.sqrt(np.array([30,100,300,1000])))
            axs[0].set_xticklabels(map("$\sqrt{{{:d}}}$".format, [30,100,300,1000]))
            axs[0].grid(ls=':')
            axs[1].set_xticks(np.sqrt(np.array([30,100,300,1000])))
            axs[1].set_xticklabels(map("$\sqrt{{{:d}}}$".format, [30,100,300,1000]))
            axs[1].grid(ls=':')


            plt.savefig(panels_dir / f'fig4B{suffix}.svg')
            plt.savefig(panels_dir / f'fig4B{suffix}.png')

            fig, axs = subplots_from_axsize(1, 2, (4,3))

            axs[0].plot(result['channel_length_sqrt'], 1/result['max_bitrate'], marker='o', color='k', ls='none')
            axs[0].plot(result['channel_length_sqrt'], predict_max_bitrate_sqrt_inv)
            axs[1].plot(result['channel_length_sqrt'], result['optimal_interval'], marker='o', color='k', ls='none')
            axs[1].plot(result['channel_length_sqrt'], predict_optimal_interval_sqrt)
            axs[0].set_xlabel('1 / sqrt(channel length)')
            axs[0].set_ylabel('max_bitrate [bit/min]')
            # axs[0].set_xlim(left=0)
            # axs[0].set_ylim(bottom=0)

            plt.savefig(panels_dir / f'fig4B{suffix}--aux3.svg')
            plt.savefig(panels_dir / f'fig4B{suffix}--aux3.png')


            fig, axs = subplots_from_axsize(1, 2, (4,3))

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


            plt.savefig(panels_dir / f'fig4B{suffix}--aux4.svg')
            plt.savefig(panels_dir / f'fig4B{suffix}--aux4.png')

