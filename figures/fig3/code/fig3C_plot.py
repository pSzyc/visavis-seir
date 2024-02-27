from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize
from itertools import product

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'approach5'
out_dir = Path(__file__).parent.parent / 'panels'

fig, ax = subplots_from_axsize(1, 1, (20,16), left=.8)
fig_propensities, axs = subplots_from_axsize(1, 2, (20,16), left=.8)
    

for channel_length, subdir_name, style in (
    (30, 'approach6', ':'),
    (100, 'approach7', '--'),
    (300, 'approach5', '-.'),
    (1000, 'approach8', '-'),
):
    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / subdir_name
    probabilities = pd.read_csv(data_dir / 'probabilities.csv').set_index('interval')
    propensities = pd.read_csv(data_dir / 'propensities.csv').set_index('interval')

    probabilities.rename(columns=lambda col: f'{channel_length} {col}', inplace=True)
    propensities.rename(columns=lambda col: f'{channel_length} {col}', inplace=True)

    probabilities.plot(marker='o', ms=3, ls=style, ax=ax)
    ax.set_ylabel('probability')
    ax.set_xlabel('interval [min]')


    probabilities.plot(marker='o', ms=3, ls=style, ax=axs[0])
    axs[0].set_ylabel('probability')
    axs[0].set_xlabel('interval [min]')

    propensities.plot(marker='o', ms=3, ls=style, ax=axs[1])
    axs[1].set_ylabel('propensity [steps$^{-1}$]')
    axs[1].set_xlabel('interval [min]')

fig.savefig(out_dir / 'fig3C.png')
fig.savefig(out_dir / 'fig3C.svg')

fig_propensities.savefig(out_dir / 'fig3C--propensities.png')
fig_propensities.savefig(out_dir / 'fig3C--propensities.svg')

