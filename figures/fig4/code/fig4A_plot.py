import pandas as pd
from matplotlib import pyplot as plt

from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import plot_scan

data_dir = Path(__file__).parent.parent / 'data'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)
suffix = "-rl25-reconstruction"

entropies = pd.read_csv(data_dir / f'fig4A_entropies1{suffix}.csv')

plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
)

plt.ylim(0,1)

plt.savefig(panels_dir / f'fig4A{suffix}.svg')
plt.savefig(panels_dir / f'fig4A{suffix}.png')
