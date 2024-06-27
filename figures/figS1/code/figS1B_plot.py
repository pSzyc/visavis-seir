# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator
from subplots_from_axsize import subplots_from_axsize
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.style import *
from scripts.defaults import PARAMETERS_DEFAULT
from scripts.entropy_utils import get_progression_var

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / "approach10"
panel_dir = Path(__file__).parent.parent / 'panels'
panel_dir.mkdir(exist_ok=True, parents=True)

fig, ax = subplots_from_axsize(1, 1, (2.5, 2))

variance_per_step = pd.read_csv(data_dir / 'variance_per_step.csv').set_index('channel_width')
variance_per_step[variance_per_step['channel_length'].eq(300) & (variance_per_step.index.get_level_values('channel_width') > 2)].plot(style='o-', ax=ax, color="C5")
variance_per_step[variance_per_step['channel_length'].eq(30) & (variance_per_step.index.get_level_values('channel_width') <= 2)].plot(style='o', fillstyle='none', ax=ax, color="C5")

analytical_variance_per_step = get_progression_var(PARAMETERS_DEFAULT, n_neighbors=2)
print(analytical_variance_per_step)

ax.set_xlim(left=0)
ax.set_ylim(bottom=0, top = 7)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(.5))
ax.hlines([analytical_variance_per_step], 0, 22, linestyles=':', color='gray')
ax.set_xlabel('channel width $W$')
ax.set_ylabel(r'transit time variance $\sigma_0^2$ [min$^{2}$/cell layer]')
ax.get_legend().set_visible(False)
ax.grid(which='both', ls=':', alpha=.4)
plt.savefig(panel_dir / 'figS1B.png')
plt.savefig(panel_dir / 'figS1B.svg')