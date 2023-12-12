from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from subplots_from_axsize import subplots_from_axsize
from sklearn.linear_model import LinearRegression
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.defaults import PARAMETERS_DEFAULT, MOL_STATES_DEFAULT


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / 'approach5'
out_dir = Path(__file__).parent.parent / 'panels'
out_dir.mkdir(exist_ok=True, parents=True)


velocities = pd.read_csv(data_dir / 'velocities.csv')

fig, ax = subplots_from_axsize(1,1,(3,3))
ax.plot(velocities['channel_width'], 60 * velocities['velocity'], marker='o', lw=1, ms=3)
analytical_velocity = 60 / (MOL_STATES_DEFAULT['n_e'] / PARAMETERS_DEFAULT['e_incr'] + .5 / PARAMETERS_DEFAULT['c_rate'])
ax.hlines([analytical_velocity], 0, 20, color='k', alpha=.4, ls=':')
ax.set_xlim(0, 21)
ax.set_ylim(0, 21)
ax.set_xlabel('channel width')
ax.set_ylabel('velocity [steps/h]')


plt.savefig(out_dir / 'figS1.svg')
plt.savefig(out_dir / 'figS1.png')



plt.show()