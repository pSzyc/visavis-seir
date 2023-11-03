from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize

data_dir = Path(__file__).parent.parent / 'data'
out_dir = Path(__file__).parent.parent / 'panels'

data = pd.read_csv(data_dir / 'fig3B--propensities.csv').set_index('interval')

fig, ax = subplots_from_axsize(1, 1, (3,2.5))

data.plot(y=['l', 'l_failure', 'l_spawning'], ax=ax)
plt.legend([
    'total event propensity ',
    'failure propensity',
    'spawning propensity',
    ])
plt.yscale('log')

plt.savefig(out_dir / 'fig2C-log.png')
plt.savefig(out_dir / 'fig2C-log.svg')

fig, ax = subplots_from_axsize(1, 1, (3,2.5))

data.plot(y=['l', 'l_failure', 'l_spawning'], ax=ax)
plt.legend([
    'total ',
    'propagation failure',
    'additional front spawning',
    ])
plt.ylim(0,0.0005)
plt.ylabel('propensity [steps$^{-1}$]')
plt.xlabel('channel width') 

plt.savefig(out_dir / 'fig2C.png')
plt.savefig(out_dir / 'fig2C.svg')

