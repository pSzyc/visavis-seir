# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk

 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import xlogy, erf

from pathlib import Path
from subplots_from_axsize import subplots_from_axsize

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.style import *
from scripts.binary import plot_scan
from scripts.entropy_utils import get_efficiency_from_extinction_probab
from scripts.handler_tuple_vertical import HandlerTupleVertical

LOG2 = np.log(2)

channel_lengths = [30,100,300,1000]


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4AB' / 'approach7'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)

fig, ax = subplots_from_axsize(1, 1, (3.0, 2.8), top=.2, left=0.5, wspace=0.5, right=0.1)

 
entropies = pd.read_csv(data_dir / f'fig4AB_entropies-c25.csv')
entropies_cm = pd.read_csv(data_dir / f'fig4AB_entropies-cm25.csv')
entropies_cmp = pd.read_csv(data_dir / f'fig4AB_entropies-cmp25.csv')
entropies_cmm = pd.read_csv(data_dir / f'fig4AB_entropies-cmm25.csv')
avg_n_backward = 1.285

plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    ms=3,
    ax=ax,
)

plot_scan(
    entropies_cm, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    alpha=0.3,
    ms=3,
    ax=ax,
    marker='s',
    fillstyle='none',
)

plot_scan(
    entropies_cmp, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    alpha=0.3,
    ms=3,
    ax=ax,
    marker='^',
    fillstyle='none',
)

plot_scan(
    entropies_cmm, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    alpha=0.3,
    ms=3,
    ax=ax,
    marker='v',
    fillstyle='none',
)



ax.set_ylim(0,1)
ax.set_xlabel('interval between slots $T_{\\mathrm{slot}}$ [min]')
handles = ax.get_legend().legend_handles
ax.legend(
    handles=list(zip(handles[:4], handles[5:9])) + [tuple(handles[:4]), tuple(handles[5:9]), tuple(handles[10:14]),tuple(handles[15:19]), handles[4]], #title='channel length $L$',
    labels=list(map('L = {:d}'.format, channel_lengths)) + ['nearest', 'nearest and 1 preceeding', 'nearest, 1 preceeding and 1 following', 'nearest and 2 preceeding', 'perfect'],
    handler_map={tuple: HandlerTupleVertical(nrows=1, vpad=-2.3)},
    )

fig.savefig(panels_dir / f'figS9-with_legend.svg')
fig.savefig(panels_dir / f'figS9.png')

ax.get_legend().set_visible(False)
fig.savefig(panels_dir / f'figS9.svg')
