import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from subplots_from_axsize import subplots_from_axsize

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.style import *


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / 'fig2C' / 'approach8'
panel_dir = Path(__file__).parent.parent / 'panels'
panel_dir.mkdir(exist_ok=True, parents=True)

channel_widths = list(range(1,10)) + list(range(10,21,2))
chosen_channel_widths = [4,6,10,20]

field_forward = 'forward'
field_backward = 'backward'



fig_scatter, axs_scatter = subplots_from_axsize(2, 2, (1.2, 1.2), wspace=0.3, hspace= 0.3, left=0.4, top=0.2, right=0.02)
fig_shares, ax_shares = subplots_from_axsize(1, 1, (2.4, 2.7), left=.75, right=0.1)
fig_hist, ax_hist = subplots_from_axsize(1, 1, (2,6))


specific = pd.read_csv(data_dir / 'detailed_event_propensities.csv').set_index('channel_width')
specific.columns = [col.replace('spawned fronts', '\nspawned fronts') for col in specific.columns]

events = pd.read_csv(data_dir / 'event_counts.csv').set_index('channel_width')


# --- Fig 2D ---

for ax, (channel_width, events_part) in zip(axs_scatter.flatten(), events[events.index.get_level_values('channel_width').isin(chosen_channel_widths)].groupby('channel_width')):
    ax.scatter(events_part[field_forward], events_part[field_backward], s=300*7500.*events_part['propensity'], alpha=0.4, c='red' )
    ax.set_xlim(0 - .5, 15 + .5)
    ax.set_ylim(0 - .5, 15 + .5)
    ax.set_title(f"$W$ = {channel_width}", loc='center', pad=-40, fontweight='bold') 
    ax.plot([0,15], [0,15], color='grey', alpha=.2)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

for ax in axs_scatter[1]:
    ax.set_xlabel('# forward fronts')
for ax in axs_scatter[:, 0]:
    ax.set_ylabel('# backward fronts')
    

fig_scatter.savefig(panel_dir / 'fig2D.svg')
fig_scatter.savefig(panel_dir / 'fig2D.png')


# --- Fig 2E ---

specific.plot(marker='^', ms=4.5, ax=ax_shares, clip_on=False)

ax_shares.set_ylim(0,2e-4)
ax_shares.yaxis.set_major_locator(MultipleLocator(2e-4))
ax_shares.yaxis.set_minor_locator(MultipleLocator(1e-4))
ax_shares.yaxis.set_major_formatter(lambda x,_: f"{x*10000:.0f}×10$^{{-4}}$")
ax_shares.set_ylabel('propensity [cell layer$^{-1}$]', labelpad=-14)
ax_shares.set_xlim(0, 21)
ax_shares.set_xlabel('channel width $W$')
ax_shares.xaxis.set_major_locator(MultipleLocator(5))
ax_shares.get_lines()[0].set_color(plt.get_cmap('tab10')(3))
ax_shares.get_lines()[2].set_color(plt.get_cmap('tab10')(8))
ax_shares.get_lines()[3].set_color(plt.get_cmap('tab10')(6))
ax_shares.legend()

fig_shares.savefig(panel_dir / 'fig2E.svg')
fig_shares.savefig(panel_dir / 'fig2E.png')


# --- Fig 2D--histograms ---

for w, events_part in events.groupby('channel_width'):
    if len(events_part):
        ax_hist.bar(*list(zip(*list(events_part.groupby('total_spawned')['count'].sum().items()))), bottom=800*w, color='maroon')

ax_hist.set_ylabel('channel width $W$')
ax_hist.set_yticks([800 * w for w in channel_widths], channel_widths)
ax_hist.set_xlim(0,25)
ax_hist.set_xlabel('total spawned fronts')
ax_hist.xaxis.set_major_locator(MultipleLocator(10))

fig_hist.savefig(panel_dir / 'fig2D--histograms.svg')
fig_hist.savefig(panel_dir / 'fig2D--histograms.png')


