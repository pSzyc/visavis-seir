from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize
from itertools import product
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.style import *

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3B' / 'approach5'
out_dir = Path(__file__).parent.parent / 'panels'

probabilities = pd.read_csv(data_dir / 'probabilities.csv').set_index('interval')
propensities = pd.read_csv(data_dir / 'propensities.csv').set_index('interval')

fig, ax = subplots_from_axsize(1, 1, (3, 2), left=.5)
probabilities.plot(marker='o', ms=3, ax=ax)
# ax.vlines([67.5, 97.6], 0,1)
ax.set_ylabel('probability')
ax.set_xlabel('interval [min]')

plt.savefig(out_dir / 'fig3B.png')
plt.savefig(out_dir / 'fig3B.svg')

fig, axs = subplots_from_axsize(1, 2, (7.5,5), left=.8)
probabilities.plot(marker='o', ms=3, ax=axs[0])
axs[0].set_ylabel('probability')
axs[0].set_xlabel('interval [min]')

propensities.plot(marker='o', ms=3, ax=axs[1])
axs[1].set_ylabel('propensity [steps$^{-1}$]')
axs[1].set_xlabel('interval [min]')
axs[1].set_xscale('log')

plt.savefig(out_dir / 'fig3B--propensities.png')
plt.savefig(out_dir / 'fig3B--propensities.svg')

