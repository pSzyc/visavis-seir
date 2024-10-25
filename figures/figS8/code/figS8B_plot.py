# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

import numpy as np
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize
from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.semi_analytical import plot_predictions, get_predictions
from scripts.style import *

panels_dir = Path(__file__).parent.parent / 'panels' 
panels_dir.mkdir(parents=True, exist_ok=True)

cmap = plt.get_cmap('brg')

channel_lengths = [30, 100, 300, 1000]
sending_probabs = .1 * np.arange(1,10)
full_sending_probabs = .01 * np.arange(1,100)
intervals = np.linspace(20,280,101)

fig, axs = subplots_from_axsize(2, 2, (2.4,2.2), top=.2, hspace=.45, wspace=.3)

for ax, channel_length in zip(axs.flatten(), channel_lengths):
    for it, sending_probab in enumerate(sending_probabs):
        plot_predictions(
            intervals=intervals,
            channel_length=channel_length,
            sending_probab=sending_probab,
            prediction_types=['failure_backward_and_variance'],
            label=f"{sending_probab:.1f}",
            color=cmap(sending_probab),
            alpha=1. if abs(sending_probab - .5) < .001 else .4,
            ax=ax,
        )

    predictions = np.concatenate(
        [
            get_predictions(
                intervals=intervals,
                channel_length=channel_length,
                sending_probab=sending_probab,
                prediction_types=['failure_backward_and_variance'],
            )['failure_backward_and_variance'].reshape(1, -1)
            for sending_probab in full_sending_probabs
        ], axis=0)
    
    max_prediction = predictions.max(axis=0)
    ax.fill_between(intervals, max_prediction, 0, color='k', alpha=.2, label='opt')
    ax.set_title(f'$L$ = {channel_length}', loc='left', pad=-20, fontweight='bold')
    ax.set_ylim(0,1)
    xticks = list(ax.get_xticks())
    ax.set_xticks([intervals.min()] + xticks + [intervals.max()])
    ax.set_xlim(intervals.min(), intervals.max())


for ax in axs[:, 1]:
    ax.set_ylabel('')
    # ax.set_yticklabels([])

for ax in axs[0]:
    ax.set_xlabel('')
    # ax.set_xticklabels([])

axs[1][1].legend(title='q', title_fontproperties={'weight': 'bold'}, labelspacing=.1,)

plt.savefig(panels_dir / f'figS8B.svg')
plt.savefig(panels_dir / f'figS8B.png')

