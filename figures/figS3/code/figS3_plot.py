# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
 

# This file is based on fig2C_plot.py.

from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from subplots_from_axsize import subplots_from_axsize
import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.style import *


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3' / 'approach8'
out_dir = Path(__file__).parent.parent / 'panels'

channel_length = 300

propensities = pd.read_csv(data_dir / 'figS3--propensities.csv')
propensities = propensities[propensities['channel_length'] == channel_length]
propensities_cropped_for_plot = propensities[propensities['channel_width'] <= 7]

coefs = pd.read_csv(data_dir / f'coefs--l-{channel_length}.csv').set_index('coefficient')
a_sp = coefs['spawning']['a']
b_sp = coefs['spawning']['b']
a_fail = coefs['failure']['a']
b_fail = coefs['failure']['b']



def make_plot(ax, ylim):
    xs = np.linspace(0,20,101)
    points_to_show = propensities_cropped_for_plot['l_failure'].between(*ylim)
    ax.plot(propensities_cropped_for_plot[points_to_show]['channel_width'], propensities_cropped_for_plot[points_to_show]['l_failure'], 's', color='olive', label='propagation failure', clip_on=False)
    points_to_show = propensities['l_spawning'].between(*ylim)
    ax.plot(propensities[points_to_show]['channel_width'], propensities[points_to_show]['l_spawning'], '^', color='maroon',label='front spawning', clip_on=False)
    ax.plot(xs, np.exp(a_fail * xs.reshape(-1,1) + b_fail), color='olive', alpha=0.3, 
        label=r"$\lambda_{\mathrm{fail}}=\exp(a_{\mathrm{fail}}^'~×~(W - W_{\mathrm{fail}}^'))$",
        )
    ax.plot(xs, (a_sp * xs + b_sp).reshape(-1,1), color='maroon', alpha=.4, 
        label=r"$\lambda_{\mathrm{spawn}}=a_{\mathrm{spawn}}^'~×~(W - W_{\mathrm{spawn}}^')$",
        )
    ax.plot(xs, (np.exp(a_fail * xs + b_fail) + (a_sp * xs + b_sp)).reshape(-1,1), color='navy', alpha=.4, 
        label=r"$\lambda_{\mathrm{tot}} = \lambda_{\mathrm{fail}} + \lambda_{\mathrm{spawn}}$"),
    ax.set_xlabel('channel width $W$')
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylabel('propensity [cell layer$^{-1}$]')
    ax.set_ylim(ylim)


fig, axs = subplots_from_axsize(1, 2, (60 / 25.4, 42 / 25.4), wspace=0.6, bottom=0.4)

make_plot(axs[0], ylim=(0,6e-4*2))
axs[0].yaxis.set_major_locator(MultipleLocator(2e-4))
axs[0].yaxis.set_major_formatter(lambda x,_: f"{x*10000:.0f}×10$^{{-4}}$")

make_plot(axs[1], ylim=(1.5e-5, 1))
axs[1].set_yscale('log')
# axs[1].set_ylim(2e-5, None)
axs[1].legend()

plt.savefig(out_dir / 'figS3.png', bbox_inches="tight")
plt.savefig(out_dir / 'figS3.svg', bbox_inches="tight")

