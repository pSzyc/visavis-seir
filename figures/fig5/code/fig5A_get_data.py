from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
from sklearn.linear_model import LinearRegression
from subplots_from_axsize import subplots_from_axsize
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.defaults import PARAMETERS_DEFAULT
from scripts.propensities import get_propensities_batch

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig5' / 'fig5A' / 'approach5'
data_dir.mkdir(exist_ok=True, parents=True)

channel_length = 300
channel_widths = (list(range(1,10,1)) + list(range(10,21,2)))#[::-1]
channel_widths = [1,4,5,6,10,16,20]#[::-1]



fold_changes= [0.7, 0.9, 1.0, 1.1, 1.3]#, 1.5,]
# altered_parameters = ['e_forward_rate']
altered_parameters = ['e_forward_rate', 'i_forward_rate', 'r_forward_rate', 'c_rate']

result_parts = []

for altered_parameter, fold_change in product(altered_parameters, fold_changes):

    outdir = data_dir / altered_parameter / str(fold_change)
    outdir.mkdir(exist_ok=True, parents=True)

    parameters = PARAMETERS_DEFAULT.copy()
    parameters.update({
        altered_parameter: fold_change * parameters[altered_parameter]
    })

    v = 1.25  / (PARAMETERS_DEFAULT['e_subcompartments_count'] / parameters['e_forward_rate'] + 0.5 / parameters['c_rate'])


    propensities = get_propensities_batch(
        channel_widths=channel_widths,
        channel_lengths=[channel_length],
        n_simulations=30000,
        n_workers=20,
        interval_after=int(2.5 * channel_length / v),
        parameters=parameters,
        v=v,
        per_width_kwargs = {
            w: {
                'front_direction_minimal_distance': min(max(w - 1, 1), 5),
                'min_peak_height': 0.03 / w,
            } for w in channel_widths
        },
        results_file=outdir / 'fig2C--propensities.csv',
        use_cached=True,
        # plot_results=True,
        save_iterations=False,
        ).reset_index()

    print(propensities)

    propensities_cropped_for_spawning = propensities[propensities['l_spawning'] > 0.3 * propensities['l_failure']]
    lreg = LinearRegression().fit(propensities_cropped_for_spawning[['channel_width']], propensities_cropped_for_spawning[['l_spawning']])
    a_sp, b_sp = lreg.coef_[0,0], lreg.intercept_[0]

    propensities_cropped = propensities[propensities['channel_width'].le(6) & propensities['l_failure'].gt(0) & (propensities['l_failure'] > 0.3 * propensities['l_spawning'])]
    lreg_failure = LinearRegression().fit(propensities_cropped[['channel_width']], np.log(propensities_cropped[['l_failure']]))
    a_fail, b_fail = lreg_failure.coef_[0,0], lreg_failure.intercept_[0]



    # ---- plot fig2C for debug ----- 
    
    propensities_cropped_for_plot = propensities[propensities['channel_width'] <= 7]


    def make_plot(ax, ylim):
        xs = np.linspace(0,20,101)
        points_to_show = propensities_cropped_for_plot['l_failure'].between(*ylim)
        ax.plot(propensities_cropped_for_plot[points_to_show]['channel_width'], propensities_cropped_for_plot[points_to_show]['l_failure'], 's', color='olive', label='propagation failure', clip_on=False)
        points_to_show = propensities['l_spawning'].between(*ylim)
        ax.plot(propensities[points_to_show]['channel_width'], propensities[points_to_show]['l_spawning'], '^', color='maroon',label='front spawning', clip_on=False)
        ax.plot(xs, np.exp(a_fail * xs.reshape(-1,1) + b_fail), color='olive', alpha=0.3, 
            # label=f"$\\lambda_f$ = {np.exp(b_fail):.2f} $\\times$ {np.exp(-a_fail):.2f}$^{{-w}}$"
            # label=r"$\lambda_{\mathrm{failure}}=\exp(\alpha W + \beta)$"
            label=f"$\\lambda_{{\mathrm{{failure}}}}=\exp({a_fail:.3f}~×~(W {b_fail / a_fail:+.3f}))$",
            )
        ax.plot(xs, (a_sp * xs + b_sp).reshape(-1,1), color='maroon', alpha=.4, 
            # label=f"$\\lambda_s$ = (w - {-b_sp/a_sp:.2f}) / {1/a_sp:.0f}"
            # label=r"$\lambda_{\mathrm{spawning}}=a W + b$"
            label=f"$\\lambda_{{\mathrm{{spawning}}}}={a_sp:.3g}~×~(W {b_sp / a_sp:+.3f})$",
            )
        ax.plot(xs, (np.exp(a_fail * xs + b_fail) + (a_sp * xs + b_sp)).reshape(-1,1), color='navy', alpha=.4, 
            label=r"$\lambda_{\mathrm{tot}} = \lambda_{\mathrm{failure}} + \lambda_{\mathrm{spawning}}$"),
        ax.set_xlabel('channel width $W$')
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_ylabel('propensity [step$^{-1}$]')
        ax.set_ylim(ylim)


    fig, axs = subplots_from_axsize(1, 2, (5,5), wspace=0.6, bottom=0.4)

    make_plot(axs[0], ylim=(0,40*a_sp))
    # axs[0].yaxis.set_major_locator(MultipleLocator(2e-4))
    axs[0].yaxis.set_major_formatter(lambda x,_: f"{x*10000:.0f}×10$^{{-4}}$")

    make_plot(axs[1], ylim=(.5e-5, 1))
    axs[1].set_yscale('log')
    # axs[1].set_ylim(2e-5, None)
    axs[1].legend()

    plt.savefig(outdir / 'fig2C.png', bbox_inches="tight")
    plt.savefig(outdir / 'fig2C.svg', bbox_inches="tight")


    # ---- save results ----- 


    result_part = {
        'altered_parameter': altered_parameter,
        'fold_change': fold_change,
        'a_spawning': a_sp,
        'b_spawning': b_sp,
        'a_failure': a_fail,
        'b_failure': b_fail,
    }
    result_parts.append(result_part)

result = pd.DataFrame(result_parts).set_index(['altered_parameter', 'fold_change'])
result.to_csv(data_dir / 'fit_coefficients.csv')

