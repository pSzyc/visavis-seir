import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.style import *
data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / 'fig-1'

data = pd.read_csv(data_dir / 'pulse_fates.csv')
data = data[data['fate'] == 'transmitted']
plt.figure(figsize=(80 / 25.4, 80 / 25.4))
for channel_width in data['channel_width'].unique():
    data[data['channel_width'] == channel_width].groupby('channel_length')['track_end'].var().plot(style='o-', label="channel width = " + str(channel_width))
plt.legend()
plt.ylabel(r'variance of propagation time [min$^{2}$]')
plt.xlabel('channel length')
plt.gca().yaxis.set_major_locator(MultipleLocator(200))
plt.savefig(Path(__file__).parent.parent / 'panels' / 'figS1_plot-1.png')
plt.savefig(Path(__file__).parent.parent / 'panels' / 'figS1_plot-1.svg')