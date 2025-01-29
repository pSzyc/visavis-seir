# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
 

from pathlib import Path
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.style import *
from scripts.parameter_scan import plot_parameter_scan


data_dir_rates = Path(__file__).parent.parent.parent.parent / 'data' / 'fig6' /  'fig6A' / 'rates' / 'approach3'
data_dir_states = Path(__file__).parent.parent.parent.parent / 'data' / 'fig6' /  'fig6A' / 'states' / 'approach3'
out_dir = Path(__file__).parent.parent / 'panels'


results_rates = pd.read_csv(data_dir_rates / 'optimized_frequencies.csv').set_index(['altered_parameter', 'fold_change'])
results_states = pd.read_csv(data_dir_states / 'optimized_frequencies.csv').set_index(['altered_parameter', 'n_states'])

results_rates = results_rates[1 / results_rates['max_value'] < 1000]
results_states = results_states[1 / results_states['max_value'] < 1000]



letter = 'A'
feature_name = 'minimum average interval\nat channel end [min]'
plot_parameter_scan(1 / results_rates['max_value'], 1 / results_states['max_value'], feature_name=feature_name, ylim=(0,600), plot_states=False, color='coral')
plt.savefig(out_dir / ("fig6" + (letter or re.sub(r'\s+', '_', feature_name)) + ".svg"))
plt.savefig(out_dir / ("fig6" + (letter or re.sub(r'\s+', '_', feature_name)) + ".png"))


letter = 'A1'
feature_name = 'optimal interval $T_{\\mathrm{slot}}$ [min]'
plot_parameter_scan(results_rates['optimal_interval'], results_states['optimal_interval'], feature_name=feature_name, ylim=(0, 600))
plt.savefig(out_dir / ("fig6" + (letter or re.sub(r'\s+', '_', feature_name)) + ".svg"))
plt.savefig(out_dir / ("fig6" + (letter or re.sub(r'\s+', '_', feature_name)) + ".png"))

