import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from subplots_from_axsize import subplots_from_axsize
from sklearn.linear_model import LinearRegression
from numpy.polynomial.polynomial import Polynomial

from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import plot_scan

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)
suffix = "-c"
for fields in 'c', 'rl', 'cm', 'cp', 'cmp':
    for k_neighbors in (15, 25):
        for reconstruction in (True, False):
            suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"
            print(f"Drawing plots for suffix: {suffix}")

            entropies = pd.read_csv(data_dir / f'fig4C_entropies{suffix}.csv')


            fig, ax = subplots_from_axsize(1, 1, (10, 10))

            fig, ax = plot_scan(
                entropies, 
                c_field=['channel_length', 'channel_width'],
                x_field='interval',
                y_field='bitrate_per_hour',
                ax=ax,
            )

            expected = {
                30: 90,
                300: 150,
            }

            result_parts = []

            for it, ((channel_length, channel_width), data) in enumerate(entropies.groupby(['channel_length', 'channel_width',])):
                naive_optimal_interval = data['interval'].iloc[data['bitrate_per_hour'].argmax()]
                data = data[data['interval'].between(naive_optimal_interval - 30, naive_optimal_interval + 30)]
                # poly_fit, err_matrix = Polynomial.fit(data['interval'], data['bitrate_per_hour'], deg=2, full=True)
                # poly_fit_coefs = poly_fit.coef[::-1]
                # print(poly_fit_coefs, *err_matrix)
                poly_fit_coefs, err_matrix = np.polyfit(data['interval'], data['bitrate_per_hour'], deg=2, cov=True)
                optimal_interval = -0.5 * poly_fit_coefs[1] / poly_fit_coefs[0]
                optimal_interval_diff = np.array([0.5 * poly_fit_coefs[1] / poly_fit_coefs[0]**2, -0.5 / poly_fit_coefs[0], 0])
                optimal_interval_sqerr = ((optimal_interval_diff.reshape(-1, 1) @ optimal_interval_diff.reshape(1, -1)) * err_matrix).sum().sum()
                optimal_interval_err = np.sqrt(optimal_interval_sqerr)

                max_bitrate = poly_fit_coefs[2] - 0.25 * poly_fit_coefs[1]**2 / poly_fit_coefs[0]
                max_bitrate_diff = np.array([0.25 * poly_fit_coefs[1]**2 / poly_fit_coefs[0]**2, -0.5 * poly_fit_coefs[1] / poly_fit_coefs[0] , 1])
                max_bitrate_sqerr = ((max_bitrate_diff.reshape(-1, 1) @ max_bitrate_diff.reshape(1, -1)) * err_matrix).sum().sum()
                max_bitrate_err = np.sqrt(max_bitrate_sqerr)


                poly_fit_fun = np.poly1d(poly_fit_coefs)
                ax.plot(data['interval'], poly_fit_fun(data['interval']), '-', lw=1, color=f'C{it}')

                result_part = pd.Series({
                    'channel_width': channel_width,
                    'channel_length': channel_length,
                    'channel_length_sqrt': np.sqrt(channel_length),
                    'optimal_interval': optimal_interval,
                    'optimal_interval_sq': optimal_interval**2,
                    'optimal_interval_err': optimal_interval_err,
                    'max_bitrate': max_bitrate,
                    'max_bitrate_sq': max_bitrate**2,
                    'max_bitrate_log': np.log(max_bitrate),
                    'max_bitrate_inv': 1 / max_bitrate,
                    'max_bitrate_err': max_bitrate_err,
                })
                result_parts.append(result_part)



            result = pd.DataFrame(result_parts)
            print(result)

            plt.scatter(result['optimal_interval'], result['max_bitrate'], color='k', marker='o', s=10)

            plt.ylim(0,1)

            plt.savefig(panels_dir / f'fig4C{suffix}--aux1.svg')
            plt.savefig(panels_dir / f'fig4C{suffix}--aux1.png')


            fig, axs = subplots_from_axsize(1, 2, (4,3), top=0.2)

            ls = np.linspace(result['channel_width'].min(), result['channel_width'].max())

            for ch_l_it, (channel_length, result_group) in enumerate(result.groupby('channel_length')):
                result_group.plot.line('channel_width', 'max_bitrate', marker='o', color=f'C{ch_l_it}', yerr='max_bitrate_err', capsize=4, ls='none', ax=axs[0], label=f"{channel_length:.0f}")

            axs[0].set_ylabel('max bitrate [bit/h]')
            axs[0].set_xlim(left=0)
            axs[0].set_ylim(bottom=0, top=1)
            axs[0].legend(title='channel length')

            for ch_l_it, (channel_length, result_group) in enumerate(result.groupby('channel_length')):
                result_group.plot.line('channel_width', 'optimal_interval', marker='o', color=f'C{ch_l_it}', yerr='optimal_interval_err', capsize=4, ls='none', ax=axs[1], label=f"{channel_length:.0f}")
  
            axs[1].set_ylabel('optimal interval [min]')
            axs[1].set_xlim(left=0)
            axs[1].set_ylim(bottom=0, top=200)
            axs[1].legend(title='channel length')


            plt.savefig(panels_dir / f'fig4C{suffix}.svg')
            plt.savefig(panels_dir / f'fig4C{suffix}.png')



