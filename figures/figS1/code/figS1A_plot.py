import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib.ticker import MultipleLocator
from subplots_from_axsize import subplots_from_axsize
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.style import *

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / "approach6"
fig2C_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / "fig2C" / 'approach8'    
panel_dir = Path(__file__).parent.parent / 'panels'
panel_dir.mkdir(exist_ok=True, parents=True)
data_dir.mkdir(exist_ok=True, parents=True)

channel_length = 300
channel_widths = list(range(1,10)) + list(range(10,21,2))


# plt.figure(figsize=(80 / 25.4, 80 / 25.4))

# data = pd.concat([
#     pd.read_csv(fig2C_data_dir / f'w-{channel_width}-l-{channel_length}' / 'pulse_fates.csv')
#     for channel_width in channel_widths
#     ], ignore_index=True
# )

# data = data[data['track_end_position'] > 40]
# data['speed'] = data['track_end_position'] / data['track_end']
# data.groupby('channel_width')['speed'].mean().to_csv(data_dir / 'velocities.csv')

velocity = pd.read_csv(data_dir / 'velocity.csv').set_index('channel_width')

fig, ax = subplots_from_axsize(1, 1, (2.5, 2), left=.8)

velocity.plot(style='o-', ax=ax)

# data.groupby('channel_width')['speed'].mean().plot(style='o-')


yticks = [0.1 * y for y in range(5)]
ax.set_ylabel('front propagation speed $v$ [step/min]')
ax.set_xlabel('channel width')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0, top = 0.4)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_ticks(yticks + [1/4.5, 1/3.5], [f"{x:.1f}" for x in yticks] + [r'$v_{\mathrm{deterministic}}$', r'$v_{\mathrm{asymptotic}}$'])
ax.hlines([1/4.5], 0, 22, linestyles=':', color='gray')
ax.hlines([1/3.5], 0, 22, linestyles='--', color='gray')
ax.grid(which='both', ls=':', alpha=.4)
ax.get_legend().set_visible(False)
plt.savefig(panel_dir / 'figS1A.png')
plt.savefig(panel_dir / 'figS1A.svg')
