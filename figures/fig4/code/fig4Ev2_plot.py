import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize


import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.formula import get_factors, get_range, get_length_with_variance_equal_refractory, get_permament_spawning_site_lambda
from scripts.defaults import PARAMETERS_DEFAULT
from scripts.style import *

panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)


ww = np.linspace(1,20,101)
ll = 10**np.linspace(0,5,51)

N = 100

ranges = get_range(ww)
no_permanent_spawning_site_ranges = np.abs(1 / N / get_permament_spawning_site_lambda(ww))
equilibria = get_length_with_variance_equal_refractory(ww)

fig, ax = subplots_from_axsize(1, 1, (1.68, 1.5), left=0.5, wspace=0.5, right=0.01)

ax.plot(ranges, ww, c='k')
ax.plot(no_permanent_spawning_site_ranges[no_permanent_spawning_site_ranges <= ranges], ww[no_permanent_spawning_site_ranges <= ranges], c='k', ls=':')
ax.plot(equilibria[equilibria <= ranges], ww[equilibria <= ranges], c='k', ls='--')

ax.plot([30,  1000], [6,  6], c='k',     lw=3, alpha=0.4)
ax.plot([30,  30  ], [1, 12], c='b',     lw=3, alpha=0.4)
ax.plot([300, 300 ], [1, 12], c='green',lw=3, alpha=0.4)

ax.set_xscale('log')
ax.xaxis.set_major_formatter(lambda x,_: f"{x:.0f}")
# ax.yaxis.set_major_formatter(lambda x,_: f"{x+1:.0f}")
# ax.set_yticks([4,9,14,19])
ax.set_ylabel('channel width')
ax.set_xlabel('channel length')

plt.savefig(panels_dir / 'fig4Ev2.svg')
plt.savefig(panels_dir / 'fig4Ev2.png')