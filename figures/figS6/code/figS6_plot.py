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


channel_length_to_approach = {
    300: 'approach5',
    30: 'approach6',
    100: 'approach7',
    1000: 'approach8',
}


channel_lengths = [30,100,300,1000]


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4AB' /'approach5'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)

fig, ax = subplots_from_axsize(1, 1, (2*1.68, 2*1.5), top=.2, left=0.5, wspace=0.5, right=0.1)

 
entropies = pd.read_csv(data_dir / f'fig4AB_entropies-c25.csv')
entropies_2pts = pd.read_csv(data_dir / f'fig4AB_entropies-cm15.csv')
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
    entropies_2pts, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    alpha=0.3,
    ms=3,
    ax=ax,
)


ax.set_ylim(0,1)
ax.set_xlabel('interval between slots [min]')
handles = ax.get_legend().legend_handles
ax.legend(
    handles=list(zip(handles[:4], handles[5:9])) + [tuple(handles[:4]), tuple(handles[5:9]), handles[4]], #title='channel length $L$',
    labels=list(map('L = {:d}'.format, channel_lengths)) + ['based on 1 front', 'based on 2 fronts', 'perfect'],
    handler_map={tuple: HandlerTupleVertical(nrows=1, vpad=-2.3)},
    )

fig.savefig(panels_dir / f'figS6.svg')
fig.savefig(panels_dir / f'figS6.png')
