from pathlib import Path
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from subplots_from_axsize import subplots_from_axsize
import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.style import *
from scripts.defaults import PARAMETERS_DEFAULT
from scripts.parameter_scan import plot_parameter_scan


data_dir_rates = Path(__file__).parent.parent.parent.parent / 'data' / 'fig5' /  'fig5A' / 'approach5'
data_dir_states = Path(__file__).parent.parent.parent.parent / 'data' / 'fig5' /  'fig5BC' / 'approach1'
out_dir = Path(__file__).parent.parent / 'panels'
out_dir_figS4 = Path(__file__).parent.parent.parent / 'figS4' / 'panels'



def get_optimal_w(coefficients):
    return -(np.log(-coefficients['a_failure'] / coefficients['a_spawning']) + coefficients['b_failure']) / coefficients['a_failure'] 

def predict_total_event_propensity(width, coefficients):
    return np.exp(coefficients['a_failure'] * width + coefficients['b_failure']) + coefficients['a_spawning'] * width + coefficients['b_spawning']


def plot_fig2C(coefficients, ax, show_optimal_w=False, show_min_propensity=False, show_legend=False):
    xs = np.linspace(1, 16, 101)
    ws = np.arange(1, 17)
    ymax = 4e-4
    def clip(y):
        return np.where(y <= ymax, y, np.nan)
    
    ax.plot(xs, clip(np.exp(coefficients['a_failure'] * xs.reshape(-1,1) + coefficients['b_failure'])), color='olive', alpha=0.3, 
        label=r"$\lambda_{\mathrm{fail}}=\exp(a_{\mathrm{fail}}~×~(W - W_{\mathrm{fail}}))$",
        )

    ax.plot(xs, clip(coefficients['a_spawning'] * xs + coefficients['b_spawning']).reshape(-1,1), color='maroon', alpha=.4, 
        label=r"$\lambda_{\mathrm{spawn}}=a_{\mathrm{spawn}}~×~(W - W_{\mathrm{spawn}})$",
        )
    
    ax.plot(xs, clip((np.exp(coefficients['a_failure'] * xs + coefficients['b_failure']) + (coefficients['a_spawning'] * xs + coefficients['b_spawning']))).reshape(-1,1), color='navy', alpha=.4, 
        label=r"$\lambda_{\mathrm{tot}} = \lambda_{\mathrm{fail}} + \lambda_{\mathrm{spawn}}$",
        )


    ax.set_yscale('linear')
    if show_legend:
        ax.set_ylim(-5e-4,4e-4)
    else:
        ax.set_ylim(-5e-4, 4e-4)

    if show_min_propensity:
        ax.set_xlim(-3,24)
    else:
        ax.set_xlim(0,22)

    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))


    ax.set_xticks([])
    ax.set_yticks([])

    optimal_w = get_optimal_w(coefficients)
    min_propensity = predict_total_event_propensity(optimal_w, coefficients)

    if show_optimal_w:
        ax.plot([optimal_w,optimal_w], [0, min_propensity], color='navy', ls=':')
        ax.set_xticks([optimal_w], ['$W_{opt}$'])


    if show_min_propensity:
        ax.plot([1,optimal_w], [min_propensity, min_propensity], color='navy', ls=':')
        ax.set_yticks([min_propensity], ['min. $\\lambda_{tot}$'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if show_legend:
        ax.legend(loc='lower center', bbox_to_anchor=(.25, -.25 + (.1 if not show_optimal_w else 0) ), handlelength=1.2)
    ax.set_ylabel(r'propensity $\rightarrow$', loc='top')# if show_legend else 'center')
    ax.set_xlabel(r'channel width $\rightarrow$')



coefficients_rates = pd.read_csv(data_dir_rates / 'fit_coefficients.csv').set_index(['altered_parameter', 'fold_change'])
coefficients_states = pd.read_csv(data_dir_states / 'fit_coefficients.csv').set_index(['altered_parameter', 'n_states'])


optimal_w_rates = get_optimal_w(coefficients_rates)
optimal_w_states = get_optimal_w(coefficients_states)

minimal_propensity_rates = predict_total_event_propensity(optimal_w_rates, coefficients_rates)
minimal_propensity_states = predict_total_event_propensity(optimal_w_states, coefficients_states)

ranges_rates = predict_total_event_propensity(pd.DataFrame([list(range(1,21))] * len(coefficients_rates), index = coefficients_rates.index).T, coefficients_rates).T
ranges_states = predict_total_event_propensity(pd.DataFrame([list(range(1,21))] * len(coefficients_states), index = coefficients_states.index).T, coefficients_states).T

optimal_w_rouded_rates = ranges_rates.idxmin(axis=1) + 1
optimal_w_rouded_states = ranges_states.idxmin(axis=1) + 1

minimal_propensity_rounded_rates = ranges_rates.min(axis=1)
minimal_propensity_rounded_states = ranges_states.min(axis=1)

letter = 'A2'
feature_name = 'optimal width $W_{opt}$'
axs = plot_parameter_scan(optimal_w_rates, optimal_w_states, feature_name=feature_name)
ax = axs[1,0]
ax.set_visible(True)
plot_fig2C(coefficients_rates.loc['e_forward_rate', PARAMETERS_DEFAULT['e_forward_rate']], ax, show_optimal_w=True)
plt.savefig(out_dir / ("fig5" + (letter or re.sub(r'\s+', '_', feature_name)) + ".svg"))
plt.savefig(out_dir / ("fig5" + (letter or re.sub(r'\s+', '_', feature_name)) + ".png"))

letter = 'B2'
feature_name = 'minimal $\\lambda_{tot}$ [cell layer$^{-1}$]'
axs = plot_parameter_scan(minimal_propensity_rates, minimal_propensity_states, feature_name=feature_name, logscale=True)
ax = axs[1,0]
ax.set_visible(True)
plot_fig2C(coefficients_rates.loc['e_forward_rate', PARAMETERS_DEFAULT['e_forward_rate']], ax, show_min_propensity=True)
plt.savefig(out_dir / ("fig5" + (letter or re.sub(r'\s+', '_', feature_name)) + ".svg"))
plt.savefig(out_dir / ("fig5" + (letter or re.sub(r'\s+', '_', feature_name)) + ".png"))




letter = 'A'
feature_name = 'coefficients'
axs = plot_parameter_scan(coefficients_rates['a_failure'], coefficients_states['a_failure'], feature_name=r'$a_{\mathrm{fail}},  \log_{10} a_{\mathrm{spawn}}$', ylim=(-6.7, 0), color='olive', alpha=.7)
plot_parameter_scan(np.log10(coefficients_rates['a_spawning']), np.log10(coefficients_states['a_spawning']), feature_name=r'coefficient value', ylim=(-6.7, 0),  color='maroon', alpha=.7, axs=axs)
ax = axs[1,0]
ax.set_visible(True)
plot_fig2C(coefficients_rates.loc['e_forward_rate', PARAMETERS_DEFAULT['e_forward_rate']], ax, show_legend=True)
axs[0,0].legend()
handles, _ = axs[0,0].get_legend_handles_labels()
axs[0,0].legend(handles, [r'$a_{\mathrm{fail}}$', r'$\log_{10} a_{\mathrm{spawn}}$'], loc='lower right', ncol=2, handlelength=1)
plt.savefig(out_dir / ("fig5" + (letter or re.sub(r'\s+', '_', feature_name)) + ".svg"))
plt.savefig(out_dir / ("fig5" + (letter or re.sub(r'\s+', '_', feature_name)) + ".png"))



letter = 'B'
feature_name = 'minimal $\\lambda_{tot}$ [cell layer$^{-1}$]'
axs = plot_parameter_scan(minimal_propensity_rounded_rates, minimal_propensity_rounded_states, feature_name=feature_name, logscale=True)
ax = axs[1,0]
ax.set_visible(True)
plot_fig2C(coefficients_rates.loc['e_forward_rate', PARAMETERS_DEFAULT['e_forward_rate']], ax, show_min_propensity=True)
plt.savefig(out_dir / ("fig5" + (letter or re.sub(r'\s+', '_', feature_name)) + ".svg"))
plt.savefig(out_dir / ("fig5" + (letter or re.sub(r'\s+', '_', feature_name)) + ".png"))


letter = 'C'
feature_name = 'optimal width $W_{opt}$'
axs = plot_parameter_scan(optimal_w_rouded_rates, optimal_w_rouded_states, feature_name=feature_name, ylim=(0,11), grid=True, color='slategray')
ax = axs[1,0]
ax.set_visible(True)
plot_fig2C(coefficients_rates.loc['e_forward_rate', PARAMETERS_DEFAULT['e_forward_rate']], ax, show_optimal_w=True)
plt.savefig(out_dir / ("fig5" + (letter or re.sub(r'\s+', '_', feature_name)) + ".svg"))
plt.savefig(out_dir / ("fig5" + (letter or re.sub(r'\s+', '_', feature_name)) + ".png"))



letter = 'S4'
feature_name = 'coefficients'
axs = plot_parameter_scan(-coefficients_rates['b_failure'] / coefficients_rates['a_failure'], -coefficients_states['b_failure'] / coefficients_states['a_failure'], feature_name=r'$a_{\mathrm{fail}},  \log_{10} a_{\mathrm{spawn}}$', color='darkseagreen', alpha=1.)
plot_parameter_scan(-coefficients_rates['b_spawning'] / coefficients_rates['a_spawning'], -coefficients_states['b_spawning'] / coefficients_states['a_spawning'], feature_name='coefficient value',  color='rosybrown', alpha=1., axs=axs)
ax = axs[1,0]
ax.set_visible(True)
plot_fig2C(coefficients_rates.loc['e_forward_rate', PARAMETERS_DEFAULT['e_forward_rate']], ax, show_legend=True)
axs[0,0].legend()
handles, _ = axs[0,0].get_legend_handles_labels()
axs[0,0].legend(handles, [r'$W_{\mathrm{fail}}$', r'$W_{\mathrm{spawn}}$'], loc='lower right', ncol=2, handlelength=1)
plt.savefig(out_dir_figS4 / ('figS4' + ".svg"))
plt.savefig(out_dir_figS4 / ('figS4' + ".png"))



