from pathlib import Path
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from itertools import product
from subplots_from_axsize import subplots_from_axsize

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.style import *
from scripts.defaults import PARAMETERS_DEFAULT
from scripts.parameter_scan import plot_parameter_scan


data_dir_rates = Path(__file__).parent.parent.parent.parent / 'data' / 'fig6' /  'fig6D' / 'rates' /'approach2'
data_dir_states = Path(__file__).parent.parent.parent.parent / 'data' / 'fig6' /  'fig6D' / 'states' /'approach2'
out_dir = Path(__file__).parent.parent / 'panels'
out_dir_figS5 = Path(__file__).parent.parent.parent / 'figS5' / 'panels'


channel_lengths = [30,100,300,1000]
fold_changes = np.exp(np.linspace(-1, 1, 21))
n_states_s = np.arange(1,11)

fields = 'c'
k_neighbors = 25
reconstruction = False
suffix = f"{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"


result_rates = pd.concat([
            pd.read_csv(data_dir_rates / altered_parameter / f"{fold_change:.3f}" / suffix / f'optimized_bitrate.csv', index_col=['channel_width', 'channel_length'])
            for altered_parameter, fold_change in product(['c_rate', 'e_forward_rate', 'i_forward_rate', 'r_forward_rate'], fold_changes)
        ], names=['altered_parameter', 'fold_change'], keys=product(['c_rate', 'e_forward_rate', 'i_forward_rate', 'r_forward_rate'], fold_changes))
result_states = pd.concat([
        pd.read_csv(data_dir_states / altered_parameter / f"{n_states}" / suffix / f'optimized_bitrate.csv', index_col=['channel_width', 'channel_length'])
        for altered_parameter, n_states in product(['e_subcompartments_count', 'i_subcompartments_count', 'r_subcompartments_count'], n_states_s)
    ], names=['altered_parameter', 'n_states'], keys=product(['e_subcompartments_count', 'i_subcompartments_count', 'r_subcompartments_count'], n_states_s))
      

result_rates['max_bitrate_per_hour'] = 60 * result_rates['max_bitrate']
result_states['max_bitrate_per_hour'] = 60 * result_states['max_bitrate']


figname = 'fig6D'
feature_name = 'maximal bitrate [bit/h]'
axs = None
for channel_length in channel_lengths:
    axs = plot_parameter_scan(
        result_rates[result_rates.index.get_level_values('channel_length') == channel_length]['max_bitrate_per_hour'], 
        result_states[result_states.index.get_level_values('channel_length') == channel_length]['max_bitrate_per_hour'], 
        feature_name=feature_name, color=channel_length_to_color[channel_length], axs=axs, ylim=(0,1), plot_states=False,)
axs[0,0].legend()
handles, _ = axs[0,0].get_legend_handles_labels()
axs[0,0].legend(handles[:4], channel_lengths, title='channel length $L$')
plt.savefig(out_dir / (figname + ".svg"))
plt.savefig(out_dir / (figname + ".png"))


figname = 'figS5'
feature_name = 'maximal bitrate [bit/h]'
axs = None
for channel_length in channel_lengths:
    axs = plot_parameter_scan(
        result_rates[result_rates.index.get_level_values('channel_length') == channel_length]['max_bitrate_per_hour'], 
        result_states[result_states.index.get_level_values('channel_length') == channel_length]['max_bitrate_per_hour'], 
        feature_name=feature_name, color=channel_length_to_color[channel_length], axs=axs, ylim=(0,1), plot_rates=False,)
plt.savefig(out_dir_figS5 / (figname + ".svg"))
plt.savefig(out_dir_figS5 / (figname + ".png"))




