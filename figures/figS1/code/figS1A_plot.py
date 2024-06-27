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

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / "approach10"
panel_dir = Path(__file__).parent.parent / 'panels'
panel_dir.mkdir(exist_ok=True, parents=True)

fig, ax = subplots_from_axsize(1, 1, (2.5, 2), left=.8)

velocity = pd.read_csv(data_dir / 'velocity.csv').set_index('channel_width')
velocity[velocity['channel_length'].eq(300)].plot(style='o-', ax=ax, color="C4")
velocity[velocity['channel_length'].eq(30) & (velocity.index.get_level_values('channel_width') == 1)].plot(style='o-', fillstyle='none', ax=ax, color="C4")

yticks = [0.1 * y for y in range(5)]
ax.set_ylabel('front propagation speed $v$ [cell layer/min]')
ax.set_xlabel('channel width $W$')
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
