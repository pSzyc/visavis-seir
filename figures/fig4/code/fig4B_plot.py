import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from subplots_from_axsize import subplots_from_axsize
from sklearn.linear_model import LinearRegression

from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import plot_scan

data_dir = Path(__file__).parent.parent / 'data'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)
# suffix = "-c"
# for fields in 'c', 'rl', 'cm', 'cp', 'cmp':
for fields in 'c', 'cm':
    for k_neighbors in (15, 25):
        # for reconstruction in (True, False):
        for reconstruction in (False,):
            suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"
            print(f"Drawing plots for suffix: {suffix}")

            entropies = pd.read_csv(data_dir / f'fig4B_entropies{suffix}.csv')

            fig, ax = plot_scan(
                entropies, 
                c_field='channel_length',
                x_field='interval',
                y_field='bitrate_per_hour',
            )

            expected = {
                30: 110, 
                52: 115,
                100: 120,
                170: 130,
                300: 140,
            }

            result_parts = []

            for it, (channel_length, data) in enumerate(entropies.groupby('channel_length')):
                # data = data[data['interval'].between(expected[channel_length] - 30, expected[channel_length] + 30)]
                poly_fit = np.polyfit(data['interval'], data['bitrate_per_hour'], deg=2)
                optimal_interval = -0.5 * poly_fit[1] / poly_fit[0]
                max_bitrate = poly_fit[2] - 0.25 * poly_fit[1]**2 / poly_fit[0] 
                poly_fit_fun = np.poly1d(poly_fit)
                ax.plot(data['interval'], poly_fit_fun(data['interval']), '-', lw=1, color=f'C{it}')

                result_part = pd.Series({
                    'channel_length': channel_length,
                    'channel_length_sqrt': np.sqrt(channel_length),
                    'optimal_interval': optimal_interval,
                    'optimal_interval_sq': optimal_interval**2,
                    'max_bitrate': max_bitrate,
                    'max_bitrate_sq': max_bitrate**2,
                    'max_bitrate_log': np.log(max_bitrate),
                    'max_bitrate_inv': 1 / max_bitrate,
                })
                result_parts.append(result_part)



            result = pd.DataFrame(result_parts)
            print(result)

            plt.scatter(result['optimal_interval'], result['max_bitrate'], color='k', marker='o', s=10)

            plt.ylim(0,1)

            plt.savefig(panels_dir / f'fig4B{suffix}--aux1.svg')
            plt.savefig(panels_dir / f'fig4B{suffix}--aux1.png')



            fit_max_bitrate_sqrt_inv = LinearRegression().fit(result[['channel_length_sqrt']], result[['max_bitrate_inv']])
            predict_max_bitrate_sqrt_inv = fit_max_bitrate_sqrt_inv.predict(result[['channel_length_sqrt']])
            print(f"max_bitrate = 1 / (sqrt(l)/{1/fit_max_bitrate_sqrt_inv.coef_[0][0]:.2g} + {fit_max_bitrate_sqrt_inv.intercept_[0]:.2g})")

            fit_optimal_interval_sqrt = LinearRegression().fit(result[['channel_length_sqrt']], result[['optimal_interval']])
            predict_optimal_interval_sqrt = fit_optimal_interval_sqrt.predict(result[['channel_length_sqrt']])
            print(f"optimal_interval = {fit_optimal_interval_sqrt.coef_[0][0]:.2g}*sqrt(l) + {fit_optimal_interval_sqrt.intercept_[0]:.2g}")

            fig, axs = subplots_from_axsize(1, 2, (4,3))

            result.plot.line('channel_length_sqrt', 'max_bitrate_inv', marker='o', color='k', ls='none', ax=axs[0])
            axs[0].plot(result['channel_length_sqrt'], predict_max_bitrate_sqrt_inv)
            result.plot.line('channel_length_sqrt', 'optimal_interval', marker='o', color='k', ls='none', ax=axs[1])
            axs[1].plot(result['channel_length_sqrt'], predict_optimal_interval_sqrt)
            axs[0].set_xlabel('$\sqrt{\mathrm{channel~length}}$')
            axs[0].set_ylabel('(maximal bitrate)$^{-1}$ [h/bit]')
            axs[1].set_xlabel('$\sqrt{\mathrm{channel~length}}$')
            axs[1].set_ylabel('optimal interval [min]')
            axs[0].legend(
                ['simulations', f"$\sqrt{{L~/~{1/fit_max_bitrate_sqrt_inv.coef_[0][0]**2:.03g}}} + {fit_max_bitrate_sqrt_inv.intercept_[0]:.03g}$"],
                loc='upper left',
                )
            axs[1].legend(
                ['simulations', f"${fit_optimal_interval_sqrt.coef_[0][0]:.03g} × \sqrt{{L}} + {fit_optimal_interval_sqrt.intercept_[0]:.03g}$"],
                loc='upper left',
                )
            axs[0].set_ylim(0,5.2)
            axs[1].set_ylim(0,312)

            plt.savefig(panels_dir / f'fig4B{suffix}--aux2.svg')
            plt.savefig(panels_dir / f'fig4B{suffix}--aux2.png')

            fig, axs = subplots_from_axsize(1, 2, (4,3))

            axs[0].plot(result['channel_length_sqrt'], 1/result['max_bitrate'], marker='o', color='k', ls='none')
            axs[0].plot(result['channel_length_sqrt'], predict_max_bitrate_sqrt_inv)
            axs[1].plot(result['channel_length_sqrt'], result['optimal_interval'], marker='o', color='k', ls='none')
            axs[1].plot(result['channel_length_sqrt'], predict_optimal_interval_sqrt)
            axs[0].set_xlabel('1 / sqrt(channel length)')
            axs[0].set_ylabel('max_bitrate [bit/h]')
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
                label=f"1 / [sqrt(L) / {1/fit_max_bitrate_sqrt_inv.coef_[0][0]:.2g} + {fit_max_bitrate_sqrt_inv.intercept_[0]:.2g}]",
                )

            axs[0].set_xlim(left=0)
            axs[0].set_ylim(bottom=0, top=1)
            axs[0].legend()

            result.plot.line('channel_length', 'optimal_interval', marker='o', color='k', ls='none', ax=axs[1])
            axs[1].plot(
                ls,
                fit_optimal_interval_sqrt.predict(np.sqrt(ls).reshape(-1,1)),
                label=f"{fit_optimal_interval_sqrt.coef_[0][0]:.2g} × sqrt(L) + {fit_optimal_interval_sqrt.intercept_[0]:.2g}",
                )
            axs[1].set_xlim(left=0)
            axs[1].set_ylim(bottom=0, top=200)
            axs[1].legend()


            plt.savefig(panels_dir / f'fig4B{suffix}.svg')
            plt.savefig(panels_dir / f'fig4B{suffix}.png')

