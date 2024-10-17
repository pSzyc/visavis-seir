# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

# This file is based on fig2C_plot.py.

from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from subplots_from_axsize import subplots_from_axsize

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.style import *


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4AB' / 'approach7'
panel_dir = Path(__file__).parent.parent / 'panels'

channel_length = 300
channel_width = 6
intervals = [150, 90]

fig, axs = subplots_from_axsize(1, len(intervals), (3.5, 2.2), wspace=.3, sharey=True)

xlim = (-4.5*150, 4.5*150)


for ax, interval in zip(axs, intervals):
    data = pd.read_csv(data_dir / f'l-{channel_length}-w-{channel_width}-i-{interval}' / 'dataset.csv')

    data.groupby('x')['c+0'].plot.hist(bins=np.arange(xlim[0], xlim[1] + 1, 10), alpha=0.3, density=True, ax=ax)
    ax.legend(['$S=0$ (no pulse sent)', '$S=1$ (pulse sent)'], loc='upper right')
    ax.set_xlabel('$\Delta t$ [min]')
    ax.set_ylabel('probability')
    ax.set_yticks([])
    # ax.axvline(0, color='k', ls='--', alpha=0.3)
    for i in range(int(np.floor(xlim[0] / interval)), int(np.floor(xlim[1] / interval)) + 1):
        ax.axvline(i * interval, color='k', ls='--', alpha=0.1)
    ax.xaxis.set_major_locator(MultipleLocator(interval))
    ax.set_xlim(xlim)
    ax.set_title(f"$T_{{\\mathrm{{slot}}}}$ = {interval}", loc='left', pad=-20, fontweight='bold')


plt.savefig(panel_dir / 'figS5.png', bbox_inches="tight")
plt.savefig(panel_dir / 'figS5.svg', bbox_inches="tight")

