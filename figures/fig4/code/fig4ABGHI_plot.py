# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from subplots_from_axsize import subplots_from_axsize

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.style import *
from scripts.binary import plot_scan
from scripts.entropy_utils import get_efficiency_from_extinction_probab
from scripts.semi_analytical import plot_predictions, get_extinction_probability_distant_fronts, get_extinction_probability_free_fronts
from scripts.handler_tuple_vertical import HandlerTupleVertical


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4AB' /'approach7'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)

fig, axs = subplots_from_axsize(1, 2, (1.68, 1.5), top=.2, left=0.5, wspace=0.5, right=0.01)
 
entropies = pd.read_csv(data_dir / 'fig4AB_entropies-c25.csv')
avg_n_backward = 1.285
sigma0 = 1.65

# --- Fig 4A ---

plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    ms=3,
    ax=axs[0],
)

axs[0].set_ylim(0,1)
axs[0].set_xlabel('interval between slots $T_{\\mathrm{slot}}$ [min]')
handles = axs[0].get_legend().legend_handles
axs[0].legend(handles=handles[:5], title='channel length $L$')


# --- Fig 4B ---

plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='efficiency',
    ms=3,
    alpha=.7,
    ax=axs[1],
)

extinction_probability_distant_fronts = get_extinction_probability_distant_fronts(entropies['channel_width'], entropies['channel_length'], gamma=1)
extinction_probability_free_fronts = get_extinction_probability_free_fronts(entropies['channel_width'], entropies['channel_length'])

for it, channel_length in enumerate(entropies['channel_length'].unique()):
    axs[1].plot(
        entropies[entropies['channel_length'] == channel_length]['interval'],
        get_efficiency_from_extinction_probab(
            extinction_probability_distant_fronts[entropies['channel_length'] == channel_length]
        ),
        alpha=.4, ls='-', lw=1, color=channel_length_to_color[channel_length], label=f'distant front L={channel_length}')

axs[1].set_ylim(0,1)
axs[1].set_xlabel('interval between slots $T_{\\mathrm{slot}}$ [min]')

axs[1].legend()
handles = axs[1].get_legend().legend_handles
axs[1].legend(
    handles=[
        tuple(handles[i] for i in range(0,4)),
        tuple(handles[i] for i in range(5,9))],
    labels=[
        'colors as \nin panel A',
         'distant-front\nlimit'
        ],
    loc='lower right',
    handler_map={tuple: HandlerTupleVertical(ncols=4, vpad=-2.3)},
    # title='colors as in panel A'
    )

fig.savefig(panels_dir / f'fig4AB.svg')
fig.savefig(panels_dir / f'fig4AB.png')
plt.close(fig)

# --- Fig 4GHI ---

fig, axs = subplots_from_axsize(3, 1, (1.68, 1.5), left=0.5, top=.2, wspace=0.1, hspace=0.65, right=0.1)

channel_width = 6
channel_lengths = [100,300,1000]
for it, (ax,channel_length) in enumerate(zip(axs, channel_lengths)):

    plot_scan(
        entropies[entropies['channel_length'] == channel_length], 
        c_field='channel_length',
        x_field='interval',
        y_field='bitrate_per_hour',
        ms=3,
        ax=ax,
        color=channel_length_to_color[channel_length],
    )

    plot_predictions(
        intervals=np.linspace(20,280,101),
        channel_length=channel_length,
        sending_probab=.5,
        prediction_types=['failure_and_backward', 'failure_backward_and_variance'],
        ax=ax,
    )

    ax.set_ylim(0,1)
    ax.set_xlabel('interval between slots $T_{\\mathrm{slot}}$ [min]')
    if True:#it == 2  :
        ax.legend()
        handles = ax.get_legend().legend_handles
        ax.legend(
            handles=[handles[i] for i in [2,3,1]], 
            labels=[
                'disruptive events', 
                'disruptive events \n& transit time variance', 
                'perfect transmission',
                ],
            title=f'prediction taking into account:',
            loc='upper center')
    else:
        ax.get_legend().set(visible=False)

    if False:
        ax.set_ylabel('')
        ax.yaxis.set_ticklabels('')
    
    ax.set_title(f"channel length $L$ = {channel_length}", loc='left', pad=-20, fontweight='bold')
    
fig.savefig(panels_dir / f'fig4GHI.svg')
fig.savefig(panels_dir / f'fig4GHI.png')
