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
from scripts.style import *
from scripts.binary import plot_scan
from scripts.defaults import PARAMETERS_DEFAULT
from scripts.formula import get_time_var, predicted_optimal_interval_formula



data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4E' / 'approach1'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)


channel_length_colors = {
    30: 0,
    300: 2,
    1000: 3,
}

full_parameter_names = {
    'r': 'refractory',
    'e': 'excited',
    'i': 'inducing',
}

# fold_changes = np.exp(np.linspace(-1, 1, 21))
# fold_changes = np.exp([0])
param_values = list(range(1,11))

from scripts.style import *

for altered_parameter in ['r_subcompartments_count']:
    for fields in 'c', :
        for k_neighbors in (25,):
            for reconstruction in (False,):
                suffix = f"{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"
                # corresponding_intermediate = f"n_{altered_parameter[0]}"
                # corresponding_time = f"{full_parameter_names[altered_parameter[0]]} time [min]"
                corresponding_rate = f"{altered_parameter[0]}_forward_rate"
                print(f"Drawing plots for suffix: {suffix}")

                entropies = pd.concat([
                    pd.read_csv(data_dir / altered_parameter / f"{param_value:.3f}" / suffix / f'entropies.csv', index_col=['channel_width', 'channel_length'])
                    for param_value in param_values
                ], names=['n_states'], keys=param_values).reset_index()
                result = pd.concat([
                    pd.read_csv(data_dir / altered_parameter / f"{param_value:.3f}" / suffix / f'optimized_bitrate.csv', index_col=['channel_width', 'channel_length'])
                    for param_value in param_values
                ], names=['n_states'], keys=param_values).reset_index()
                
                result['max_bitrate_per_hour'] = 60 * result['max_bitrate']
                print(result)
                
                # result = pd.read_csv(data_dir / suffix / f'optimized_bitrate.csv')

                fig, ax = subplots_from_axsize(1, 1, (10, 10))

                fig, ax = plot_scan(
                    entropies, 
                    c_field=['channel_length', 'n_states'],
                    x_field='interval',
                    y_field='bitrate_per_hour',
                    ms=2,
                    ax=ax,
                )
                plt.scatter(result['optimal_interval'], result['max_bitrate']*60, color='k', marker='o', s=20)

                plt.ylim(0,1)
                # old:
                plt.savefig(panels_dir / f'fig4E-{suffix}--aux1.svg')
                plt.savefig(panels_dir / f'fig4E-{suffix}--aux1.png')


                fig, axs = subplots_from_axsize(1, 3, (2.75, 1.5), left=0.5, wspace=0.5)

                ls = np.linspace(result['n_states'].min(), result['n_states'].max())

                for channel_length, result_group in result.groupby('channel_length'):
                    result_group.plot.line('n_states', 'max_bitrate_per_hour', marker='o', color=f'C{channel_length_colors[channel_length]}', #yerr='max_bitrate_err', capsize=4, 
                    ls='-', lw=1, ax=axs[0], label=f"{channel_length:.0f}")

                axs[0].set_ylabel('max bitrate [bit/h]')
                axs[0].set_xlabel('n_states')
                axs[0].set_xlim(left=0)
                axs[0].set_ylim(bottom=0, top=1)
                axs[0].get_legend().set(visible=False)
                axs[0].grid(ls=':')
                

                for channel_length, result_group in result[result['max_bitrate_per_hour'] > 0.1].groupby('channel_length'):
                    result_group.plot.line('n_states', 'optimal_interval', marker='o', color=f'C{channel_length_colors[channel_length]}', #yerr='optimal_interval_err', capsize=4, 
                    ls='-', lw=1, ax=axs[1], label=f"{channel_length:.0f}")
                    axs[1].plot(
                        result_group['n_states'], 
                        [
                            predicted_optimal_interval_formula(
                                channel_width, 
                                channel_length,
                                parameters={
                                    **PARAMETERS_DEFAULT,
                                    altered_parameter: param_value,
                                    corresponding_rate: PARAMETERS_DEFAULT[corresponding_rate] * param_value / PARAMTERS_DEFAULT[altered_parameter]},
                                )
                            for channel_width, param_value in zip(result_group['channel_width'], result_group['n_states'])
                        ],
                        alpha=0.4, color='r')


                axs[1].set_ylabel('optimal interval [min]')
                axs[1].set_xlabel('n_states')
                axs[1].set_xlim(left=0)
                axs[1].set_ylim(bottom=0, top=300)
                axs[1].legend(title='channel length')
                axs[1].grid(ls=':')


                for channel_length, result_group in result.assign(ratio=lambda df: df['max_bitrate']*df['optimal_interval'])[result['max_bitrate_per_hour'] > 0.1].groupby('channel_length'):
                    result_group.plot.line('n_states', 'ratio', marker='o', color=f'C{channel_length_colors[channel_length]}', #yerr='optimal_interval_err', capsize=4, 
                    ls='-', lw=1, ax=axs[2], label=f"{channel_length:.0f}")
                    # axs[2].plot(
                        # result_group['n_states'], 
                        # [
                            # predicted_optimal_interval_formula(
                                # channel_width, 
                                # channel_length,
                                # parameters={
                                    # **PARAMETERS_DEFAULT, 
                                    # altered_parameter: param_value,
                                    # corresponding_rate: PARAMETERS_DEFAULT[corresponding_rate] * param_value / PARAMETERS_DEFAULT[altered_parameter]
                                # },
                                # )
                            # for channel_width, param_value in zip(result_group['channel_width'], result_group['n_states'])
                        # ],
                        # alpha=0.4, color='r')


                axs[2].set_ylabel('optimal interval [min]')
                axs[2].set_xlabel('n_states')
                axs[2].set_xlim(left=0)
                axs[2].set_ylim(bottom=0, top=1)
                axs[2].legend(title='channel length')
                axs[2].grid(ls=':')
                plt.savefig(panels_dir / f'fig4E.svg')
                plt.savefig(panels_dir / f'fig4E.png')
                # old:
                # plt.savefig(panels_dir / f'fig4D--{altered_parameter}-{suffix}.svg')
                # plt.savefig(panels_dir / f'fig4D--{altered_parameter}-{suffix}.png')

# plt.show()


