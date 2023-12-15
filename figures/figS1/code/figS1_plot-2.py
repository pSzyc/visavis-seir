import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib.ticker import MultipleLocator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.style import *

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / "figS1" / 'fig-2'
data_list = []
plt.figure(figsize=(80 / 25.4, 80 / 25.4))
for path, dir, files in os.walk(data_dir):
    for file in files:
        if file == 'pulse_fates.csv':
            data_list.append(pd.read_csv(os.path.join(path, file)))
data = pd.concat(data_list)
data = data[data['track_end_position'] > 40]
data['speed'] = data['track_end_position'] / data['track_end']
data.groupby('channel_width')['speed'].mean().plot(style='o-')
plt.ylabel('front propagation speed [step/min]')
plt.xlabel('channel width')
plt.xlim(left=0)
plt.ylim(bottom=0, top = 0.4)
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
plt.hlines([1/4.5], 0, 22, linestyles='--', color='gray') 
plt.savefig(Path(__file__).parent.parent / 'panels' / 'figS1_plot-2.png')
plt.savefig(Path(__file__).parent.parent / 'panels' / 'figS1_plot-2.svg')