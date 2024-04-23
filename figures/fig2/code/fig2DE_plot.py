import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from itertools import product

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.style import *
from sklearn.linear_model import LinearRegression

from subplots_from_axsize import subplots_from_axsize


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / 'fig2C' / 'approach8'
out_dir = Path(__file__).parent.parent / 'panels'

channel_widths = list(range(1,10)) + list(range(10,21,2))
chosen_channel_widths = [4,6,10,20]
channel_length = 300
n_simulations = 30000

#fig, axs = subplots_from_axsize((len(chosen_channel_widths)-1) // 4 + 1, 4, ( 28/25.4, 28/25.4), wspace=.45)
fig_scatter, axs_scatter = subplots_from_axsize(2, 2, (1.2, 1.2), wspace=0.3, hspace= 0.3, left=0.4, top=0.2, right=0.02)
fig_shares, ax_shares = subplots_from_axsize(1, 1, (2.3, 2.7), left=.75, right=0.1)
fig_hist, ax_hist = subplots_from_axsize(1, 1, (2,6))

field_forward = 'forward'
field_backward = 'backward'

counts_selected_parts = {}


for w in channel_widths:

    first_split_events = pd.read_csv(data_dir / f'w-{w}-l-{channel_length}/first_split_events.csv').set_index(['channel_width', 'channel_length', 'simulation_id'])

    counts = (
        first_split_events[[field_forward, field_backward]]
            .reindex(list(product([w], [channel_length], range(n_simulations))), fill_value=0)
            .value_counts([field_forward, field_backward])
            .reset_index()
            .assign(
                all_spawned=lambda df: df[field_forward] + df[field_backward],
                channel_width=w,
            )
            .rename(columns={0: 'count'})
    )
    counts_selected_part = counts[counts[field_forward].gt(0) | counts[field_backward].gt(0)]
    counts_selected_parts.update({w: counts_selected_part})

for it, (ax, w) in enumerate(zip(axs_scatter.flatten(), chosen_channel_widths)):
    counts_selected = counts_selected_parts[w]
    ax.scatter(counts_selected[field_forward], counts_selected[field_backward], s=7500.*counts_selected['count']/n_simulations, alpha=0.4, c='red' )
    ax.set_xlabel('# forward fronts' if it in [2, 3] else '')
    ax.set_ylabel('# backward fronts' if it in [0, 2] else '')
    ax.set_xlim(0 - .5, 15 + .5)
    ax.set_ylim(0 - .5, 15 + .5)
    ax.set_title(f"$W$ = {w}", loc='center', pad=-40, fontweight='bold') #, y=.8
    # ax.set_title(f"$L$ = {channel_length}", loc='left', pad=-20, fontweight='bold')
    ax.plot([0,15], [0,15], color='grey', alpha=.2)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    expected_forward_front_count = (counts_selected[field_forward] * counts_selected['count']).sum() / counts_selected['count'].sum()
    expected_backward_front_count = (counts_selected[field_backward] * counts_selected['count']).sum() / counts_selected['count'].sum()
    print(w, expected_forward_front_count, expected_backward_front_count)

counts_selected_all = pd.concat(counts_selected_parts)


specific = pd.DataFrame({
        '1 backward':          counts_selected_all[counts_selected_all[field_forward].eq(0) & counts_selected_all[field_backward].eq(1)].groupby('channel_width')['count'].sum(),
        '1 forward, 1 backward':          counts_selected_all[counts_selected_all[field_forward].eq(1) & counts_selected_all[field_backward].eq(1)].groupby('channel_width')['count'].sum(),
        '1 forward':          counts_selected_all[counts_selected_all[field_forward].eq(1) & counts_selected_all[field_backward].eq(0)].groupby('channel_width')['count'].sum(),
        'other events with $\\leq$ 6 \n spawned fronts':   counts_selected_all[counts_selected_all['all_spawned'].le(6) & (counts_selected_all[field_forward].gt(1) | counts_selected_all[field_backward].gt(1))].groupby('channel_width')['count'].sum(),
        'events with > 6 \n spawned fronts':          counts_selected_all[counts_selected_all['all_spawned'].gt(6)].groupby('channel_width')['count'].sum(),
    })

(specific / n_simulations / channel_length).plot(marker='^', ms=4.5, ax=ax_shares, clip_on=False)
(specific / n_simulations / channel_length).to_csv(data_dir / 'specific.csv')
ax_shares.set_ylim(0,2e-4)
ax_shares.yaxis.set_major_locator(MultipleLocator(1e-4))
ax_shares.yaxis.set_major_formatter(lambda x,_: f"{x*10000:.0f}×10$^{{-4}}$")
ax_shares.set_ylabel('propensity [step$^{-1}$]')
ax_shares.set_xlim(0, 21)
ax_shares.set_xlabel('channel width $W$')
ax_shares.xaxis.set_major_locator(MultipleLocator(5))
ax_shares.legend()


# ax_shares.scatter([w] * (len(specific)+1), specific.tolist() + [counts_selected['count'].sum() - specific.sum()], c=range(len(specific)+1), vmin=0, vmax=len(specific))

for w, counts_selected in counts_selected_parts.items():
    if len(counts_selected):
        ax_hist.bar(*list(zip(*list(counts_selected.groupby('all_spawned')['count'].sum().items()))), bottom=800*w, color='maroon')

ax_hist.set_ylabel('channel width $W$')
ax_hist.set_yticks([800 * w for w in channel_widths], channel_widths)
ax_hist.set_xlim(0,25)
ax_hist.set_xlabel('total spawned fronts')
ax_hist.xaxis.set_major_locator(MultipleLocator(10))

fig_scatter.savefig(out_dir / 'fig2D.svg')
fig_scatter.savefig(out_dir / 'fig2D.png')

fig_shares.savefig(out_dir / 'fig2E.svg')
fig_shares.savefig(out_dir / 'fig2E.png')

fig_hist.savefig(out_dir / 'fig2D--histograms.svg')
fig_hist.savefig(out_dir / 'fig2D--histograms.png')




