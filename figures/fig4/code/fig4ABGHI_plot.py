import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import xlogy, erf

from pathlib import Path
from subplots_from_axsize import subplots_from_axsize

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.style import *
from scripts.binary import plot_scan
from scripts.entropy_utils import get_efficiency_from_extinction_probab
from scripts.handler_tuple_vertical import HandlerTupleVertical
from scripts.analyze_velocity import get_velocity
from scripts.defaults import PARAMETERS_DEFAULT

LOG2 = np.log(2)

def xlogx(x):
    return xlogy(x, x) / LOG2


channel_length_to_approach = {
    300: 'approach5',
    30: 'approach6',
    100: 'approach7',
    1000: 'approach8',
}


channel_lengths = [100,300,1000]


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4AB' /'approach7'
fig3_data_dir = lambda channel_length: Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3B' / channel_length_to_approach[channel_length]
fig2_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / 'fig2C' / 'approach8'
velocity_cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'velocity'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)

fig, axs = subplots_from_axsize(1, 2, (1.68, 1.5), top=.2, left=0.5, wspace=0.5, right=0.01)

 
entropies = pd.read_csv(data_dir / 'fig4AB_entropies-c25.csv')
avg_n_backward = 1.285

# --- Fig 4A ---

plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    ms=3,
    ax=axs[0],
)

axs[0].set_ylim(0,1)
axs[0].set_xlabel('interval between slots $T_{\\mathrm{slot}}$ [min]')
handles = axs[0].get_legend().legend_handles
axs[0].legend(handles=handles[:5], title='channel length $L$')


# --- Fig 4B ---

plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='efficiency',
    ms=3,
    alpha=.7,
    ax=axs[1],
)


coefs = pd.read_csv(fig2_data_dir / 'coefs--l-300.csv').set_index('coefficient')


failure_propensity = np.exp(coefs['failure']['a'] * entropies['channel_width'] + coefs['failure']['b'])
spawning_propensity = (coefs['spawning']['a'] * entropies['channel_width'] + coefs['spawning']['b']) 
gamma = 1 # (1 -  sum(2**(-i) * (i * 1/3.6/2 * entropies['interval'] / entropies['channel_length']).clip(0,1.) for i in range(10)) ) # chance that a backward front collides before reaching channel start

extinction_probability_distant_fronts = 1 - np.exp(-entropies['channel_length'] * (
    failure_propensity + gamma * avg_n_backward * spawning_propensity
))

extinction_probability_free_fronts = 1 - np.exp(-entropies['channel_length'] * failure_propensity)


for it, channel_length in enumerate(entropies['channel_length'].unique()):
    axs[1].plot(
        entropies[entropies['channel_length'] == channel_length]['interval'],
        get_efficiency_from_extinction_probab(
            extinction_probability_distant_fronts[entropies['channel_length'] == channel_length]
        ),
        alpha=.4, ls='-', lw=1, color=channel_length_to_color[channel_length], label=f'distant front L={channel_length}')

axs[1].set_ylim(0,1)
axs[1].set_xlabel('interval between slots $T_{\\mathrm{slot}}$ [min]')

axs[1].legend()
handles = axs[1].get_legend().legend_handles
axs[1].legend(
    handles=[
        tuple(handles[i] for i in range(0,4)),
        tuple(handles[i] for i in range(5,9))],
    labels=[
        'colors as \nin panel A',
         'distant-front\nlimit'
        ],
    loc='lower right',
    handler_map={tuple: HandlerTupleVertical(ncols=4, vpad=-2.3)},
    # title='colors as in panel A'
    )


fig.savefig(panels_dir / f'fig4AB.svg')
fig.savefig(panels_dir / f'fig4AB.png')
plt.close(fig)

# --- Fig 4GHI ---


probabilities = pd.concat([pd.read_csv(fig3_data_dir(channel_length) / 'probabilities.csv').set_index('interval') for channel_length in channel_lengths], names=['channel_length'], keys=channel_lengths)
packed_data = pd.concat([pd.read_csv(fig3_data_dir(channel_length) / 'packed_data.csv').set_index(['fate', 'interval']) for channel_length in channel_lengths], names=['channel_length'], keys=channel_lengths)


def get_measured_spawning_probability(channel_length):
    probabilities_for_given_channel_length = probabilities[probabilities.index.get_level_values('channel_length') == channel_length].reset_index('channel_length')
    return probabilities_for_given_channel_length['1 backward front spawning'] + probabilities_for_given_channel_length['other front spawning events'] #/ (1 - probabilities_for_given_channel_length['annihilation by backward front'])

def get_measured_failure_probability(channel_length):
    probabilities_for_given_channel_length = probabilities[probabilities.index.get_level_values('channel_length') == channel_length].reset_index('channel_length')
    return probabilities_for_given_channel_length['initiation failure'] + probabilities_for_given_channel_length['propagation failure'] #/ (1 - probabilities_for_channel_length['annihilation by backward front'])
   

def get_expected_number_of_backward_fronts(x, channel_length):
    return avg_n_backward * np.interp(x, *zip(*get_measured_spawning_probability(channel_length).items()))

def get_failure_probability(x, channel_length):
    return np.interp(x, *zip(*get_measured_failure_probability(channel_length).items()))



def expected_number_of_fronts_measured_from_probab(channel_length):
    # assert channel_length == selected_channel_length
    data_per_interval = packed_data[packed_data.index.get_level_values('channel_length') == channel_length].groupby('interval').sum().fillna(0)
    return data_per_interval[['fronts_forward_sum', 'fronts_backward_sum']].div(data_per_interval['count'], axis=0)
   


def get_expected_number_of_forward_fronts_from_probab(x, channel_length, sigma):
    return (1-erf((x-150) / np.sqrt(2 * channel_length * sigma)))/2 * np.interp(x, *zip(*expected_number_of_fronts_measured_from_probab(channel_length)['fronts_forward_sum'].items())) # * (-np.tanh((x - 135) / 30) + 1) / 2

def expected_number_of_backward_fronts_from_probab(x, channel_length):
    return np.interp(x, *zip(*expected_number_of_fronts_measured_from_probab(channel_length)['fronts_backward_sum'].items()))

 

def compute_mi_from_error_probabilities(sending_probab=.5, chance_for_missing=0., chance_for_fake=0.):
    tp = sending_probab * ((1 - chance_for_missing) + chance_for_missing * chance_for_fake)
    fn = sending_probab * chance_for_missing * (1 - chance_for_fake)
    fp = (1 - sending_probab) * chance_for_fake
    tn = (1 - sending_probab) * (1 - chance_for_fake)
    return (
        (xlogx(tp) + xlogx(fn) + xlogx(fp) + xlogx(tn)) # -H(S,R)
        - (xlogx(sending_probab) + xlogx(1 - sending_probab)) # -H(S)
        - (xlogx(tp + fp) + xlogx(tn + fn)) # -H(R)
    )


def chance_of_not_hitting_the_start(x, channel_length):
    return (1 - x * v / (2*channel_length)).clip(0., 1.)


def get_inaccurate_probability(interval, sigma):
    'Computes the probability that |Z| > interval for Z ~ N(0, sigma**2)'
    return 1 - erf(interval / 2 / (sigma * np.sqrt(2)))


def weighted(fn, x, channel_length, n_terms=6, **kwargs):
    'Computes the weighted sum 2**(-k) * fn(k*x)'
    return (
        sum(2**(-k) * fn(k * x, channel_length, **kwargs) for k in range(1, n_terms)) 
        + 2**(-n_terms + 1) * fn(n_terms * x, channel_length, **kwargs)
    )

fig, axs = subplots_from_axsize(3, 1, (1.68, 1.5), left=0.5, top=.2, wspace=0.1, hspace=0.65, right=0.1)

intervals = np.linspace(20,280,101)

channel_width = 6
for it, (ax,channel_length) in enumerate(zip(axs, channel_lengths)):

    v = get_velocity(channel_width, channel_length, parameters=PARAMETERS_DEFAULT, velocity_cache_dir=velocity_cache_dir)
    sigma = np.sqrt(1.65*channel_length)

    plot_scan(
        entropies[entropies['channel_length'] == channel_length], 
        c_field='channel_length',
        x_field='interval',
        y_field='bitrate_per_hour',
        ms=3,
        ax=ax,
        color=channel_length_to_color[channel_length],
    )

    inaccurate_probability = get_inaccurate_probability(intervals, sigma=sigma) #sum((-1)**k * (1-erf((2*k+1) * intervals / 2 / (sigma * np.sqrt(2)))) for k in range(5))
    failure_probability = weighted(get_failure_probability, intervals, channel_length)
    asymptotic_failure_probability = get_failure_probability(300 + 0*intervals, channel_length)
    number_of_forward_fronts = weighted(get_expected_number_of_forward_fronts_from_probab, intervals, channel_length, sigma=sigma)#, effective_mean_interval=effective_mean_interval)
    number_of_backward_fronts = weighted(get_expected_number_of_backward_fronts, intervals, channel_length)#, effective_mean_interval=effective_mean_interval)
    weighted_gamma = weighted(chance_of_not_hitting_the_start, intervals, channel_length) # chance that a backward front collides before reaching channel start

    extinction_probability = 1 - (1 - failure_probability) / (1 + number_of_backward_fronts * weighted_gamma) # probability that propagation fails or the front gets annihilated
    # asymptotic_extinction_probability = 1 - (1 - asymptotic_failure_probability) / (1 + asymptotic_number_of_backward_fronts * weighted_gamma)
    
    ax.plot(intervals, 60 * compute_mi_from_error_probabilities(
        chance_for_missing=extinction_probability,
        chance_for_fake=0.,#1-np.exp(-number_of_forward_fronts),
        ) / intervals,
        color=f"black", alpha=0.3, ls=':', label='prediction\n(failure & back-fronts)')
    # ax.plot(intervals, 60 * compute_mi_from_error_probabilities(
    #     chance_for_missing=failure_probability + (1 - failure_probability) * inaccurate_probability,
    #     chance_for_fake=1-np.exp(-number_of_forward_fronts) + np.exp(-number_of_forward_fronts)*(lambda x: 2*x-x**2)(0.5 * (1 - failure_probability) * inaccurate_probability),
    #     ) / intervals,
    #     color=f"black", alpha=0.3, ls=':', label='prediction \n(failure & variance)')
    # ax.plot(intervals, 60 * compute_mi_from_error_probabilities(
    #     chance_for_missing=asymptotic_extinction_probability + (1 - asymptotic_extinction_probability) * inaccurate_probability,
    #     chance_for_fake=1-np.exp(-asymptotic_number_of_forward_fronts) + np.exp(-asymptotic_number_of_forward_fronts)*(lambda x: 2*x-x**2)(0.5 * (1 - asymptotic_extinction_probability) * inaccurate_probability),
    #     ) / intervals,
    #     color=f"black", alpha=0.3, ls=':', label='prediction \n(asymptotic & variance)')
    ax.plot(intervals, 60 * compute_mi_from_error_probabilities(
        chance_for_missing=extinction_probability + (1 - extinction_probability) * inaccurate_probability,
        chance_for_fake=(lambda x: 2*x-x**2)(0.5 * (1 - extinction_probability) * inaccurate_probability),
        ) / intervals,
        color=f"black", alpha=0.3, ls='--', label='prediction \n(failure & back-fronts & variance)')
    # ax.plot(intervals, 60 * compute_mi_from_error_probabilities(
        # chance_for_missing=extinction_probability + (1 - extinction_probability) * inaccurate_probability,
        # chance_for_fake=number_of_forward_fronts /(1 + number_of_backward_fronts * weighted_gamma) + (lambda x: 2*x-x**2)(0.5 * (1 - extinction_probability) * inaccurate_probability),
        # ) / intervals,
        # color=f"black", alpha=0.3, ls='-', label='prediction \n(failure & back-fronts & variance)')
    
    # ax.plot(intervals, 60 * compute_mi_from_error_probabilities(
    #     chance_for_missing=extinction_probability + (1 - extinction_probability) * inaccurate_probability,
    #     chance_for_fake=1-np.exp(-number_of_forward_fronts) + np.exp(-number_of_forward_fronts)*(lambda x: 2*x-x**2)(0.5 * (1 - extinction_probability) * inaccurate_probability),
    #     ) / intervals,
    #     color=f"black", alpha=0.3, ls='-', label='prediction \n(failure & back-fronts & variance & forw-fronts)')

    ax.set_ylim(0,1)
    ax.set_xlabel('interval between slots $T_{\\mathrm{slot}}$ [min]')
    if True:#it == 2  :
        ax.legend()
        handles = ax.get_legend().legend_handles
        ax.legend(
            handles=[handles[i] for i in [2,3,1]], 
            labels=[
                'disruptive events', 
                # 'free front effects & variance', 
                'disruptive events \n& transit time variance', 
                'perfect transmission',
                ],
            title=f'prediction taking into account:',
            loc='upper center')
    else:
        ax.get_legend().set(visible=False)
        # pass

    if False:
        ax.set_ylabel('')
        ax.yaxis.set_ticklabels('')
    
    ax.set_title(f"channel length $L$ = {channel_length}", loc='left', pad=-20, fontweight='bold')
        
    
fig.savefig(panels_dir / f'fig4GHI.svg')
fig.savefig(panels_dir / f'fig4GHI.png')
