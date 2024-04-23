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


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4AB' /'approach5'
data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4AB' /'approach6'
fig3_data_dir = lambda channel_length: Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3B' / channel_length_to_approach[channel_length]
fig2_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / 'fig2C' / 'approach8'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)

fig, axs = subplots_from_axsize(1, 2, (1.68, 1.5), top=.2, left=0.5, wspace=0.5, right=0.01)

 
entropies = pd.read_csv(data_dir / f'fig4AB_entropies-c25.csv')
entropies_2pts = pd.read_csv(data_dir / f'fig4AB_entropies-cm15.csv')
avg_n_backward = 1.285

plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    ms=3,
    ax=axs[0],
)

# plot_scan(
#     entropies_2pts, 
#     c_field='channel_length',
#     x_field='interval',
#     y_field='bitrate_per_hour',
#     alpha=0.3,
#     ms=3,
#     ax=axs[0],
# )

axs[0].set_ylim(0,1)
axs[0].set_xlabel('interval between slots [min]')
handles = axs[0].get_legend().legend_handles
axs[0].legend(handles=handles[:5], title='channel length $L$')




plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='efficiency',
    ms=3,
    alpha=.7,
    ax=axs[1],
)


coefs = pd.read_csv(fig2_data_dir / 'coefs.csv').set_index('Unnamed: 0')

extinction_probability = 1-np.exp(-entropies['channel_length'] * (
    np.exp(
        coefs['failure']['a'] * entropies['channel_width'] + coefs['failure']['b'])
        + 
            avg_n_backward * (coefs['spawning']['a'] * entropies['channel_width'] + coefs['spawning']['b']) 
            # * (1 -  sum(2**(-i) * (i * 1/3.6/2 * entropies['interval'] / entropies['channel_length']).clip(0,1.) for i in range(10)) )
))

extinction_probability_only_failure = 1-np.exp(-entropies['channel_length'] * (
    np.exp(
        coefs['failure']['a'] * entropies['channel_width'] + coefs['failure']['b'])
        # + 
            # avg_n_backward * (coefs['spawning']['a'] * entropies['channel_width'] + coefs['spawning']['b']) 
            # * (1 -  sum(2**(-i) * (i * 1/3.6/2 * entropies['interval'] / entropies['channel_length']).clip(0,1.) for i in range(10)) )
))


for it, channel_length in enumerate(entropies['channel_length'].unique()):
    axs[1].plot(
        entropies[entropies['channel_length'] == channel_length]['interval'],
        get_efficiency_from_extinction_probab(
            extinction_probability[entropies['channel_length'] == channel_length]
        ),
        alpha=.4, ls='-', lw=1, color=channel_length_to_color[channel_length], label=f'distant front L={channel_length}')
    # axs[1].plot(
    #     entropies[entropies['channel_length'] == channel_length]['interval'],
    #     get_efficiency_from_extinction_probab(
    #         extinction_probability_only_failure[entropies['channel_length'] == channel_length]
    #     ),
    #     alpha=.3, ls=':', lw=1, color=channel_length_to_color[channel_length])

axs[1].set_ylim(0,1)
axs[1].set_xlabel('interval between slots [min]')
# axs[1].legend(handles=handles[:len(handles)//2], title='channel length')
# axs[1].legend(title='channel length')
# # axs[1].get_legend().set_visible(False)

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

# axs[1].get_legend().set_visible(False)

fig.savefig(panels_dir / f'fig4AB.svg')
fig.savefig(panels_dir / f'fig4AB.png')
plt.close(fig)


# entropies = pd.read_csv(data_dir / f'fig4AB_entropies-c25.csv')


# channel_lengths = entropies['channel_length'].drop_duplicates().tolist()

# propensities = pd.read_csv(fig3_data_dir / f'propensities.csv').set_index('interval')
probabilities = pd.concat([pd.read_csv(fig3_data_dir(channel_length) / f'probabilities.csv').set_index('interval') for channel_length in channel_lengths], names=['channel_length'], keys=channel_lengths)
packed_data = pd.concat([pd.read_csv(fig3_data_dir(channel_length) / f'packed_data.csv').set_index(['fate', 'interval']) for channel_length in channel_lengths], names=['channel_length'], keys=channel_lengths)
# expected_fronts = packed_data[['fronts_forward_sum', 'fronts_backward_sum']].div(packed_data['count'], axis=0).fillna(0)

# def expected_number_of_fronts_measured(channel_length, effective_mean_interval):
#     # effective_mean_interval = probabilities['initiation failure'].index * 2 / (1 - probabilities['initiation failure'])
#     # print(effective_mean_interval)
#     total_event_propensity = propensities['propagation failure'] + propensities['1 backward front spawning'] + propensities['other front spawning events']
#     return (
#         (
#             expected_fronts.loc['1 backward front spawning'].mul(propensities['1 backward front spawning'], axis=0)
#             + expected_fronts.loc['other front spawning events'].mul(propensities['other front spawning events'], axis=0)
#         ).mul(
#             (1 - probabilities['initiation failure']) # not failed at start
#             * (1 - np.exp(-channel_length * propensities['propagation failure'])) / (channel_length * propensities['propagation failure']).where(propensities['propagation failure'] > 0, 0.) # not failed along the channel
#         , axis=0,
#         )
#     )


def probability_spawning_measured(channel_length):
    probabilities_for_given_channel_length = probabilities.loc[channel_length]
    data_per_interval = (probabilities_for_given_channel_length['1 backward front spawning'] + probabilities_for_given_channel_length['other front spawning events']) #/ (1 - probabilities_for_given_channel_length['annihilation by backward front'])
    return data_per_interval

def expected_number_of_backward_fronts(x, channel_length):
    return avg_n_backward * np.interp(x, *zip(*probability_spawning_measured(channel_length).items()))



def expected_number_of_fronts_measured_from_probab(channel_length):
    # assert channel_length == selected_channel_length
    data_per_interval = packed_data[packed_data.index.get_level_values('channel_length') == channel_length].groupby('interval').sum().fillna(0)
    return data_per_interval[['fronts_forward_sum', 'fronts_backward_sum']].div(data_per_interval['count'], axis=0)
   

# def expected_number_of_forward_fronts(x, channel_length, effective_mean_interval):
#     return np.interp(x, *zip(*expected_number_of_fronts_measured(channel_length, effective_mean_interval)['fronts_forward_sum'].items()))

# def expected_number_of_backward_fronts(x, channel_length, effective_mean_interval, v=1/3.6):
    # return (channel_length - 0.5 * effective_mean_interval * v) * np.interp(x, *zip(*expected_number_of_fronts_measured(channel_length, effective_mean_interval)['fronts_backward_sum'].items()))


def expected_number_of_forward_fronts_from_probab(x, channel_length):
    return (1-erf((x-150) / np.sqrt(2 * channel_length * 1.65)))/2 * np.interp(x, *zip(*expected_number_of_fronts_measured_from_probab(channel_length)['fronts_forward_sum'].items())) # * (-np.tanh((x - 135) / 30) + 1) / 2

def expected_number_of_backward_fronts_from_probab(x, channel_length):
    return np.interp(x, *zip(*expected_number_of_fronts_measured_from_probab(channel_length)['fronts_backward_sum'].items()))


def failure_probability_measured(channel_length):
    # return probabilities['initiation failure'] + (1 - probabilities['initiation failure']) * (1 - np.exp(-channel_length * (propensities['propagation failure'])))
    # assert channel_length == selected_channel_length
    probabilities_for_channel_length = probabilities[probabilities.index.get_level_values('channel_length') == channel_length].reset_index('channel_length')
    return probabilities_for_channel_length['initiation failure'] + probabilities_for_channel_length['propagation failure'] #/ (1 - probabilities_for_channel_length['annihilation by backward front'])
    
def failure_probability(x, channel_length):
    return np.interp(x, *zip(*failure_probability_measured(channel_length).items()))


# def total_failure_probability_measured(channel_length):
#     return probabilities['initiation failure'] + (1 - probabilities['initiation failure']) * (1 - np.exp(-channel_length * (propensities['propagation failure'] + propensities['1 backward front spawning'] + propensities['other front spawning events'])))
    
# def total_failure_probability(x, channel_length):
#     return np.interp(x, *zip(*total_failure_probability_measured(channel_length).items()))
 

# def fail_to_start_probability(x, channel_length):
#     return np.interp(x, *zip(*probabilities['initiation failure'].items()))


def mi_from_error_probabilities(sending_probab=.5, chance_for_missing=0., chance_for_fake=0.):
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


def get_inaccurate_probability(xx, channel_length, sigma):
    return 1 - erf(xx / 2 / (sigma * np.sqrt(2)))

def weighted(fn, xx, channel_length, max_gap, **kwargs):
    return sum(2**(-k) * fn(k * xx, channel_length, **kwargs) for k in range(1, max_gap)) + 2**(-max_gap + 1) * fn(max_gap * xx, channel_length, **kwargs)

fig, axs = subplots_from_axsize(3, 1, (1.68, 1.5), left=0.5, top=.2, wspace=0.1, hspace=0.65, right=0.1)
# axs = axs.flatten()
# fig.delaxes(axs[2])


xx = np.linspace(20,280,101)
v = 1 / 3.6
max_gap = 6
channel_width = 6
for it, (ax,channel_length) in enumerate(zip(axs, channel_lengths)):

    plot_scan(
        entropies[entropies['channel_length'] == channel_length], 
        c_field='channel_length',
        x_field='interval',
        y_field='bitrate_per_hour',
        ms=3,
        ax=ax,
        color=channel_length_to_color[channel_length],
    )

    # plot_scan(
    #     entropies_2pts[entropies_2pts['channel_length'] == channel_length], 
    #     c_field='channel_length',
    #     x_field='interval',
    #     y_field='bitrate_per_hour',
    #     alpha=0.3,
    #     ms=3,
    #     ax=ax,
    #     color=channel_length_to_color[channel_length],
    # )


    sigma = np.sqrt(1.65*channel_length)
    inaccurate_probability = get_inaccurate_probability(xx, channel_length, sigma=sigma) #sum((-1)**k * (1-erf((2*k+1) * xx / 2 / (sigma * np.sqrt(2)))) for k in range(5))
    # weighted_total_failure_probability = weighted(total_failure_probability, xx, channel_length, max_gap)
    weighted_failure_probability = weighted(failure_probability, xx, channel_length, max_gap)
    # asymptotic_total_failure_probability = total_failure_probability(300 + 0*xx, channel_length)
    asymptotic_failure_probability = failure_probability(300 + 0*xx, channel_length)
    # effective_mean_interval = xx * 2 / (1 - weighted_total_failure_probability)
    weighted_number_of_forward_fronts = weighted(expected_number_of_forward_fronts_from_probab, xx, channel_length, max_gap)#, effective_mean_interval=effective_mean_interval)
    weighted_number_of_backward_fronts = weighted(expected_number_of_backward_fronts, xx, channel_length, max_gap)#, effective_mean_interval=effective_mean_interval)
    # asymptotic_number_of_forward_fronts = expected_number_of_forward_fronts_from_probab(300 + 0*xx, channel_length)#, effective_mean_interval=effective_mean_interval)
    # asymptotic_number_of_backward_fronts = expected_number_of_backward_fronts_from_probab(300 + 0*xx, channel_length)#, effective_mean_interval=effective_mean_interval)

    # asymptotic_number_of_backward_fronts = channel_length * (
    #     # np.exp(coefs['failure']['a'] * entropies['channel_width'] + coefs['failure']['b'])
    #     + (coefs['spawning']['a'] * channel_width + coefs['spawning']['b']) + 0 * xx
    # )


    weighted_chance_of_not_hitting_the_start = weighted(chance_of_not_hitting_the_start, xx, channel_length, max_gap)

    failure_or_anihilated_probability = 1 - (1 - weighted_failure_probability) / (1 + weighted_number_of_backward_fronts * weighted_chance_of_not_hitting_the_start)
    # asymptotic_failure_or_anihilated_probability = 1 - (1 - asymptotic_failure_probability) / (1 + asymptotic_number_of_backward_fronts * weighted_chance_of_not_hitting_the_start)
    ax.plot(xx, 60 * mi_from_error_probabilities(
        chance_for_missing=failure_or_anihilated_probability,
        chance_for_fake=0.,#1-np.exp(-weighted_number_of_forward_fronts),
        ) / xx,
        color=f"black", alpha=0.3, ls=':', label='prediction\n(failure & back-fronts)')
    # ax.plot(xx, 60 * mi_from_error_probabilities(
    #     chance_for_missing=weighted_failure_probability + (1 - weighted_failure_probability) * inaccurate_probability,
    #     chance_for_fake=1-np.exp(-weighted_number_of_forward_fronts) + np.exp(-weighted_number_of_forward_fronts)*(lambda x: 2*x-x**2)(0.5 * (1 - weighted_failure_probability) * inaccurate_probability),
    #     ) / xx,
    #     color=f"black", alpha=0.3, ls=':', label='prediction \n(failure & variance)')
    # ax.plot(xx, 60 * mi_from_error_probabilities(
    #     chance_for_missing=asymptotic_failure_or_anihilated_probability + (1 - asymptotic_failure_or_anihilated_probability) * inaccurate_probability,
    #     chance_for_fake=1-np.exp(-asymptotic_number_of_forward_fronts) + np.exp(-asymptotic_number_of_forward_fronts)*(lambda x: 2*x-x**2)(0.5 * (1 - asymptotic_failure_or_anihilated_probability) * inaccurate_probability),
    #     ) / xx,
    #     color=f"black", alpha=0.3, ls=':', label='prediction \n(asymptotic & variance)')
    ax.plot(xx, 60 * mi_from_error_probabilities(
        chance_for_missing=failure_or_anihilated_probability + (1 - failure_or_anihilated_probability) * inaccurate_probability,
        chance_for_fake=(lambda x: 2*x-x**2)(0.5 * (1 - failure_or_anihilated_probability) * inaccurate_probability),
        ) / xx,
        color=f"black", alpha=0.3, ls='--', label='prediction \n(failure & back-fronts & variance)')
    # ax.plot(xx, 60 * mi_from_error_probabilities(
        # chance_for_missing=failure_or_anihilated_probability + (1 - failure_or_anihilated_probability) * inaccurate_probability,
        # chance_for_fake=weighted_number_of_forward_fronts /(1 + weighted_number_of_backward_fronts * weighted_chance_of_not_hitting_the_start) + (lambda x: 2*x-x**2)(0.5 * (1 - failure_or_anihilated_probability) * inaccurate_probability),
        # ) / xx,
        # color=f"black", alpha=0.3, ls='-', label='prediction \n(failure & back-fronts & variance)')
    
    # ax.plot(xx, 60 * mi_from_error_probabilities(
    #     chance_for_missing=failure_or_anihilated_probability + (1 - failure_or_anihilated_probability) * inaccurate_probability,
    #     chance_for_fake=1-np.exp(-weighted_number_of_forward_fronts) + np.exp(-weighted_number_of_forward_fronts)*(lambda x: 2*x-x**2)(0.5 * (1 - failure_or_anihilated_probability) * inaccurate_probability),
    #     ) / xx,
    #     color=f"black", alpha=0.3, ls='-', label='prediction \n(failure & back-fronts & variance & forw-fronts)')
    print(pd.DataFrame({
        # 'weighted_total_failure_probability': weighted_total_failure_probability,
        'weighted_failure_probability': weighted_failure_probability,
        'weighted_number_of_backward_fronts': weighted_number_of_backward_fronts,
        'weighted_number_of_forward_fronts': weighted_number_of_forward_fronts,
        'fake_due_to_inaccuracy': (lambda x: 2*x-x**2)(0.5 * (1 - failure_or_anihilated_probability) * inaccurate_probability),
        # 'expected_number_of_backward_fronts': expected_number_of_backward_fronts(xx, channel_length, effective_mean_interval),
        # 'expected_number_of_forward_fronts': expected_number_of_forward_fronts(xx, channel_length, effective_mean_interval),
        },
        index=xx).tail(20))
    ax.set_ylim(0,1)
    ax.set_xlabel('interval between slots [min]')
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
        
    
        # ax.legend()
        # handles = ax.get_legend().legend_handles
        # ax.legend(
        #     handles=[handles[i] for i in [0,4,5,6,1]], 
        #     labels=[f'simulation', 'prediction (failure & back-fronts)', 'prediction (asymptotic & variance)', 'prediction (failure & back-fronts & variance)', 'perfect'],
        #     title=f'channel length = {channel_length}',
        #     loc='upper right')

fig.savefig(panels_dir / f'fig4GHI.svg')
fig.savefig(panels_dir / f'fig4GHI.png')
