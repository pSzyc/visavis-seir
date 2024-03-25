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


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / 'fig2C' / 'approach8'
out_dir = Path(__file__).parent.parent / 'panels'

propensities = pd.read_csv(data_dir / 'fig2C--propensities.csv')


lreg = LinearRegression().fit(propensities[['channel_width']].to_numpy(), propensities[['l_spawning']].to_numpy())
a_sp, b_sp = lreg.coef_[0,0], lreg.intercept_[0]

propensities_cropped = propensities[propensities['channel_width'] <= 6]
propensities_cropped_for_plot = propensities[propensities['channel_width'] <= 7]
lreg_failure = LinearRegression().fit(propensities_cropped[['channel_width']].to_numpy(), np.log(propensities_cropped[['l_failure']].to_numpy()))
a_fail, b_fail = lreg_failure.coef_[0,0], lreg_failure.intercept_[0]

pd.DataFrame({'spawning': {'a': a_sp, 'b': b_sp}, 'failure': {'a': a_fail, 'b': b_fail}}).to_csv(data_dir / 'coefs.csv')


def make_plot(ax, ylim):
    xs = np.linspace(0,20,101)
    points_to_show = propensities_cropped_for_plot['l_failure'].between(*ylim)
    ax.plot(propensities_cropped_for_plot[points_to_show]['channel_width'], propensities_cropped_for_plot[points_to_show]['l_failure'], 's', color='olive', label='propagation failure', clip_on=False)
    points_to_show = propensities['l_spawning'].between(*ylim)
    ax.plot(propensities[points_to_show]['channel_width'], propensities[points_to_show]['l_spawning'], '^', color='maroon',label='front spawning', clip_on=False)
    ax.plot(xs, np.exp(lreg_failure.predict(xs.reshape(-1,1))), color='olive', alpha=0.3, 
        # label=f"$\\lambda_f$ = {np.exp(b_fail):.2f} $\\times$ {np.exp(-a_fail):.2f}$^{{-w}}$"
        label=r"$\lambda_{\mathrm{failure}}=\exp(\alpha W + \beta)$"
        )
    ax.plot(xs, lreg.predict(xs.reshape(-1,1)), color='maroon', alpha=.4, 
        # label=f"$\\lambda_s$ = (w - {-b_sp/a_sp:.2f}) / {1/a_sp:.0f}"
        label=r"$\lambda_{\mathrm{spawning}}=a W + b$"
        )
    ax.plot(xs, np.exp(lreg_failure.predict(xs.reshape(-1,1))) + lreg.predict(xs.reshape(-1,1)), color='navy', alpha=.4, 
        label=r"$\lambda_{\mathrm{tot}} = \lambda_{\mathrm{failure}} + \lambda_{\mathrm{spawning}}$")
    ax.set_xlabel('channel width')
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylabel('propensity [step$^{-1}$]')
    ax.set_ylim(ylim)



# f"l_spawning = (w - {-b_sp/a_sp:.2f}) / {1/a_sp:.0f}"
# f"l_failure =  1/({np.exp(-a_fail):.2f}^(w + {b_fail/a_fail:.3f})"

fig, axs = subplots_from_axsize(1, 2, (60 / 25.4, 42 / 25.4), wspace=0.6, bottom=0.4)

make_plot(axs[0], ylim=(0,6e-4))
axs[0].yaxis.set_major_locator(MultipleLocator(2e-4))
axs[0].yaxis.set_major_formatter(lambda x,_: f"{x*10000:.0f}×10$^{{-4}}$")

make_plot(axs[1], ylim=(1.5e-5, 1))
axs[1].set_yscale('log')
# axs[1].set_ylim(2e-5, None)
axs[1].legend()

plt.savefig(out_dir / 'fig2C.png', bbox_inches="tight")
plt.savefig(out_dir / 'fig2C.svg', bbox_inches="tight")

