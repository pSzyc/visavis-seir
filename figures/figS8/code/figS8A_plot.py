# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

import numpy as np
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize
from pathlib import Path
from scipy.special import xlogy

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.style import *

def xlog2x(x):
    return xlogy(x, x) / np.log(2)

def mi(q, p):
    return xlog2x(q*p) - xlog2x(q) - xlog2x(1-q+p*q)


panels_dir = Path(__file__).parent.parent / 'panels' 
panels_dir.mkdir(parents=True, exist_ok=True)


qq = np.linspace(0, 1, 101)
fig, axs = subplots_from_axsize(1, 3, (1.7, 1.4), wspace=.45)

# MI vs q
ax = axs[0]
ps = np.linspace(0, .9, 10)

cmap = plt.get_cmap('viridis')
for p in ps:
    ax.plot(qq, mi(qq, p), label=f"{p:.1f}", color=cmap(p))
ax.set_xlabel('probability of sending a front $q$')
ax.set_ylabel('MI per slot [bit]')
ax.legend(title='extinction \nprobability $p$', labelspacing=.1, borderpad=.2)#, ncols=2)

# maximum MI vs p
ax = axs[1]
ps = np.linspace(0, 1, 11)

ax.plot(ps, mi(qq.reshape(1, -1), ps.reshape(-1, 1)).max(axis=1), label='maximum bitrate', marker='o', ms=3, alpha=.6, color='mediumslateblue')
ax.plot(ps, mi(.5, ps), label='bitrate for $q$ = 1/2', marker='o',  ms=3, alpha=.6, color='goldenrod')
ax.set_xlabel('extinction probability $p$')
ax.set_ylabel('MI per slot [bit]')
ax.legend()

# optimal q vs p
ax = axs[2]
ps = np.linspace(0, .9, 10)

ax.plot(ps, qq[mi(qq.reshape(1, -1), ps.reshape(-1, 1)).argmax(axis=1)], label='optimal $q$', marker='o',  ms=3, alpha=.6, color='mediumslateblue')
ax.set_xlabel('extinction probability $p$')
ax.set_ylabel('probability of sending a front $q$')
ax.set_ylim(axs[1].get_ylim())
ax.legend()

plt.savefig(panels_dir / f'figS8A.svg')
plt.savefig(panels_dir / f'figS8A.png')
