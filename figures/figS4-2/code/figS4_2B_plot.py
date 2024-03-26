from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from scipy.special import lambertw
from subplots_from_axsize import subplots_from_axsize

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS4-2' /  'figS4-2B' / 'approach1'
out_dir = Path(__file__).parent.parent / 'panels'

coefficients = pd.read_csv(data_dir / 'fit_coefficients.csv').set_index(['altered_parameter', 'n_states'])



# n_statess= [0.7, 0.9, 1., 1.1, 1.3]
# altered_parameters = ['i_forward_rate']
# altered_parameters = ['e_forward_rate', 'i_forward_rate', 'r_forward_rate', 'c_rate']

for altered_parameter, n_states in coefficients.index:

    coefs = coefficients.loc[altered_parameter, n_states]

    propensities = pd.read_csv(data_dir / altered_parameter / str(n_states) / 'fig2C--propensities.csv').fillna(0)

    # lreg = LinearRegression().fit(propensities[['channel_width']], propensities[['l_spawning']])
    a_sp, b_sp = coefs['a_spawning'], coefs['b_spawning']

    propensities_cropped = propensities[propensities['channel_width'].le(6) & propensities['l_failure'].gt(0) ]
    # lreg_failure = LinearRegression().fit(propensities_cropped[['channel_width']], np.log(propensities_cropped[['l_failure']]))
    a_fail, b_fail = coefs['a_failure'], coefs['b_failure']



    def make_plot(ax):
        xs = np.linspace(0,20,101)
        ax.plot(propensities_cropped['channel_width'], propensities_cropped['l_failure'], 's', color='olive', label='propagation failure')
        ax.plot(propensities['channel_width'], propensities['l_spawning'], '^', color='maroon',label='additional front spawning')
        # ax.plot(xs, np.exp(lreg_failure.predict(xs.reshape(-1,1))), color='olive', alpha=0.3, label=f"$\\lambda_f$ = {np.exp(b_fail):.2f} $\\times$ {np.exp(-a_fail):.2f}$^{{-w}}$")
        ax.plot(xs, np.exp(coefs['a_failure'] * xs + coefs['b_failure']), color='olive', alpha=0.3, label=f"$\\lambda_f$ = {np.exp(b_fail):.2f} $\\times$ {np.exp(-a_fail):.2f}$^{{-w}}$", ls=':')
        # ax.plot(xs, lreg.predict(xs.reshape(-1,1)), color='maroon', alpha=.4, label=f"$\\lambda_s$ = (w - {-b_sp/a_sp:.2f}) / {1/a_sp:.0f}")
        ax.plot(xs, coefs['a_spawning'] * xs + coefs['b_spawning'], color='maroon', alpha=0.3, label=f"$\\lambda_f$ = {np.exp(b_fail):.2f} $\\times$ {np.exp(-a_fail):.2f}$^{{-w}}$", ls=':')
        ax.plot(xs, np.exp(coefs['a_failure'] * xs + coefs['b_failure']) + coefs['a_spawning'] * xs + coefs['b_spawning'], color='navy', alpha=.4, label=f"$\\lambda_f$ + $\\lambda_s$")
        ax.set_xlabel('channel width')
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_ylabel('propensity [step$^{-1}$]')


    # f"l_spawning = (w - {-b_sp/a_sp:.2f}) / {1/a_sp:.0f}"
    # f"l_failure =  1/({np.exp(-a_fail):.2f}^(w + {b_fail/a_fail:.3f})"

    fig, axs = subplots_from_axsize(1, 2, (3,3), left=1)

    make_plot(axs[0])
    axs[0].set_ylim(0,1.2*propensities['l_spawning'].max())
    # axs[0].yaxis.set_major_locator(MultipleLocator(2e-4))
    # axs[0].yaxis.set_major_formatter(lambda x,_: f"${x*10000:.0f} \\times 10^{{-4}}$")

    make_plot(axs[1])
    axs[1].set_yscale('log')
    axs[1].set_ylim(2e-5, None)
    axs[1].legend()

    plt.savefig(data_dir / altered_parameter / str(n_states) / 'fig2C.png')
    plt.savefig(data_dir / altered_parameter / str(n_states) / 'fig2C.svg')
    plt.close(fig)





fig, axs = subplots_from_axsize(2, 2, (3,3), left=1., wspace=1.)

axs = axs.flatten()

for altered_parameter, data in coefficients.groupby('altered_parameter'):
    for ax, coef_name in zip(axs, data.columns):
        ax.plot(data.index.get_level_values('n_states'), data[coef_name] if 'a_' in coef_name else -data[coef_name] / data[coef_name.replace('b_', 'a_')], 'o-', label=altered_parameter)
        ax.set_xlabel('n_states')
        ax.set_ylabel(coef_name if 'a_' in coef_name else f"-{coef_name} / {coef_name.replace('b_', 'a_')}")
        ax.legend()
        if 'spawning' in coef_name:
            ax.set_yscale('log')

plt.savefig(out_dir / 'figS4-2B.svg')
plt.savefig(out_dir / 'figS4-2B.png')

equal_reasons_w = -1./coefficients['a_failure'] * lambertw(-coefficients['a_failure'] / coefficients['a_spawning'] * np.exp(-coefficients['a_failure'] / coefficients['a_spawning'] * coefficients['b_spawning'] + coefficients['b_failure'])) - coefficients['b_spawning'] / coefficients['a_spawning']
    
fig, ax = subplots_from_axsize(1,1 , (3,3), left=1., wspace=1.)
for altered_parameter, data in equal_reasons_w.groupby('altered_parameter'):
    ax.plot(data.index.get_level_values('n_states'), data, 'o-', label=altered_parameter)

plt.legend()
plt.savefig(out_dir / 'figS4-2B-equal_reasons_width.svg')
plt.savefig(out_dir / 'figS4-2B-equal_reasons_width.png')


optimal_w = -(np.log(-coefficients['a_failure'] / coefficients['a_spawning']) + coefficients['b_failure']) / coefficients['a_failure'] 
    
fig, ax = subplots_from_axsize(1,1 , (3,3), left=1., wspace=1.)
for altered_parameter, data in optimal_w.groupby('altered_parameter'):
    ax.plot(data.index.get_level_values('n_states'), data, 'o-', label=altered_parameter)

plt.legend()
plt.savefig(out_dir / 'figS4-2B-optimal_width.svg')
plt.savefig(out_dir / 'figS4-2B-optimal_width.png')


def predict_total_event_propensity(width):
    return np.exp(coefficients['a_failure'] * width + coefficients['b_failure']) + coefficients['a_spawning'] * width + coefficients['b_spawning']

minimal_event_propensity = predict_total_event_propensity(optimal_w)
    

fig, ax = subplots_from_axsize(1,1 , (3,3), left=1., wspace=1.)
for altered_parameter, data in minimal_event_propensity.groupby('altered_parameter'):
    ax.plot(data.index.get_level_values('n_states'), data, 'o-', label=altered_parameter)

plt.legend()
plt.savefig(out_dir / 'figS4-2B-minimal_event_propensity.svg')
plt.savefig(out_dir / 'figS4-2B-minimal_event_propensity.png')


event_propensity_for_w_6 = predict_total_event_propensity(6)

fig, ax = subplots_from_axsize(1,1 , (3,3), left=1., wspace=1.)
for altered_parameter, data in event_propensity_for_w_6.groupby('altered_parameter'):
    ax.plot(data.index.get_level_values('n_states'), 1/data, 'o-', label=altered_parameter)

plt.legend()
plt.savefig(out_dir / 'figS4-2B-event_propensity_for_w_6.svg')
plt.savefig(out_dir / 'figS4-2B-event_propensity_for_w_6.png')


