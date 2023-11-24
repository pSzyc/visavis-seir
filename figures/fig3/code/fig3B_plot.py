from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize
from itertools import product

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'approach5'
out_dir = Path(__file__).parent.parent / 'panels'

probabilities = pd.read_csv(data_dir / 'probabilities.csv').set_index('interval')
propensities = pd.read_csv(data_dir / 'propensities.csv').set_index('interval')


fig, axs = subplots_from_axsize(1, 2, (5,4), left=.8)
probabilities.plot(marker='o', ms=3, ax=axs[0])
axs[0].set_ylabel('probability')
axs[0].set_xlabel('interval [min]')

# plt.savefig(out_dir / 'fig3B.png')
# plt.savefig(out_dir / 'fig3B.svg')

# fig, ax = subplots_from_axsize(1, 1, (5,4), left=.8)
propensities.plot(marker='o', ms=3, ax=axs[1])
axs[1].set_ylabel('propensity [steps$^{-1}$]')
axs[1].set_xlabel('interval [min]')

plt.savefig(out_dir / 'fig3B--propensities.png')
plt.savefig(out_dir / 'fig3B--propensities.svg')

