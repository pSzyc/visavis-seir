import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from sklearn.linear_model import LinearRegression

from subplots_from_axsize import subplots_from_axsize


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / 'fig2C' / 'approach6'
out_dir = Path(__file__).parent.parent / 'panels'

channel_widths = list(range(1,10)) + list(range(10,21,2))
chosen_channel_widths = [4,6,8,10,20]

fig, axs = subplots_from_axsize((len(chosen_channel_widths)-1) // 5 + 1, 5, (1.5, 1.5), wspace=.45)

fig_main, ax_main = subplots_from_axsize(1, 1, (2,1.5), left=1., right=3)
fig_hist, ax_hist = subplots_from_axsize(1, 1, (2,6))

field_forward = 'reached_end'
field_backward = 'reached_start'

counts_selected_parts = {}

for w in channel_widths:

    pulse_fates = pd.read_csv(data_dir / f'w-{w}-l-300/pulse_fates.csv')
    pulse_fates['reached_end'] = pulse_fates['reached_end'] - 1

    counts = pulse_fates[pulse_fates['fate'] == 'transmitted'].value_counts([field_forward, field_backward]).reset_index().assign(
        all_spawned=lambda df: df[field_forward] + df[field_backward],
        channel_width=w,
        ).rename(columns={0:'count'})
    counts_selected_part = counts[counts[field_forward].gt(0) | counts[field_backward].gt(0)]
    counts_selected_parts.update({w: counts_selected_part})

for it, (ax, w) in enumerate(zip(axs.flatten(), chosen_channel_widths)):
    counts_selected = counts_selected_parts[w]
    ax.scatter(counts_selected[field_forward], counts_selected[field_backward], s=counts_selected['count']/4, alpha=0.4)
    ax.set_xlabel('# forward fronts')
    ax.set_ylabel('# backward fronts' if not it else '')
    ax.set_xlim(0 - .5, 13 + .5)
    ax.set_ylim(0 - .5, 18 + .5)
    ax.set_title(f"{w = }", y=.8)
    ax.plot([0,13], [0,13], color='grey', alpha=.2)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))


counts_selected_all = pd.concat(counts_selected_parts)


specific = pd.DataFrame({
        '1 backward':          counts_selected_all[counts_selected_all[field_forward].eq(0) & counts_selected_all[field_backward].eq(1)].groupby('channel_width')['count'].sum(),
        '1 forward, 1 backward':          counts_selected_all[counts_selected_all[field_forward].eq(1) & counts_selected_all[field_backward].eq(1)].groupby('channel_width')['count'].sum(),
        '1 forward':          counts_selected_all[counts_selected_all[field_forward].eq(1) & counts_selected_all[field_backward].eq(0)].groupby('channel_width')['count'].sum(),
        'other with $\\leq$ 6 spawned fronts':   counts_selected_all[counts_selected_all['all_spawned'].le(6) & (counts_selected_all[field_forward].gt(1) | counts_selected_all[field_backward].gt(1))].groupby('channel_width')['count'].sum(),
        'total with > 6 spawned fronts':          counts_selected_all[counts_selected_all['all_spawned'].gt(6)].groupby('channel_width')['count'].sum(),
    })

(specific / len(pulse_fates)).plot(marker='^', ax=ax_main)
ax_main.set_ylabel('probability')
ax_main.set_xlim(0, 21)
ax_main.set_xlabel('channel width')
ax_main.xaxis.set_major_locator(MultipleLocator(5))
ax_main.legend(loc='center left', bbox_to_anchor=(1,.5))


# ax_main.scatter([w] * (len(specific)+1), specific.tolist() + [counts_selected['count'].sum() - specific.sum()], c=range(len(specific)+1), vmin=0, vmax=len(specific))

for w, counts_selected in counts_selected_parts.items():
    if len(counts_selected):
        ax_hist.bar(*list(zip(*list(counts_selected.groupby('all_spawned')['count'].sum().items()))), bottom=800*w, color='maroon')

ax_hist.set_ylabel('channel width')
ax_hist.set_yticks([800 * w for w in channel_widths], channel_widths)
ax_hist.set_xlim(0,25)
ax_hist.set_xlabel('total spawned fronts')
ax_hist.xaxis.set_major_locator(MultipleLocator(10))

fig.savefig(out_dir / 'fig2E--scatters.svg')
fig.savefig(out_dir / 'fig2E--scatters.png')

fig_main.savefig(out_dir / 'fig2E--shares.svg')
fig_main.savefig(out_dir / 'fig2E--shares.png')

fig_hist.savefig(out_dir / 'fig2E--histograms.svg')
fig_hist.savefig(out_dir / 'fig2E--histograms.png')




