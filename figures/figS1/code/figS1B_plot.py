import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator
from subplots_from_axsize import subplots_from_axsize
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.style import *

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / 'figS1B' / "approach1"
fig2C_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / "fig2C" / 'approach8'    
panel_dir = Path(__file__).parent.parent / 'panels'
panel_dir.mkdir(exist_ok=True, parents=True)
data_dir.mkdir(exist_ok=True, parents=True)


MM = 25.4

channel_length = 300
channel_widths = list(range(2,10)) + list(range(10,21,2))


data = pd.concat([
    pd.read_csv(fig2C_data_dir / f'w-{channel_width}-l-{channel_length}' / 'pulse_fates.csv', usecols=['channel_width', 'track_end', 'fate'])
    for channel_width in channel_widths
    ], ignore_index=True
)
data = data[data['fate'] == 'transmitted']
# plt.figure(figsize=(80 / MM, 80 / MM))
fig, ax = subplots_from_axsize(1, 1, (2.5, 2))

variance_per_step = data.groupby('channel_width')['track_end'].var() / channel_length
variance_per_step.name = 'variance_per_step'
variance_per_step.to_csv(data_dir  / 'variance_per_step.csv')
variance_per_step = pd.read_csv(data_dir  / 'variance_per_step.csv').
variance_per_step.plot(style='o-')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0, top = 5)
ax.xaxis.set_major_locator(MultipleLocator(5))
# ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(.5))
ax.set_xlabel('channel width')
ax.set_ylabel(r'traveling time variance $\sigma_0$ [min$^{2}$/step]')
ax.grid(which='both', ls=':', alpha=.4)
plt.savefig(panel_dir / 'figS1B.png')
plt.savefig(panel_dir / 'figS1B.svg')