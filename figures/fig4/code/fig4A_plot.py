import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import shutil
from scipy.special import xlogy, erf

from pathlib import Path
from subplots_from_axsize import subplots_from_axsize

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.binary import plot_scan

LOG2 = np.log(2)

def xlogx(x):
    return xlogy(x, x) / LOG2

# data_dir = Path(__file__).parent.parent / 'data'
data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4A' /'approach3'
fig3_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'approach6'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)

# for fields in 'c', 'rl', 'cm', 'cp', 'cmp':
#     for k_neighbors in (15, 25):
#         for reconstruction in (True, False):

#             suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"

#             entropies = pd.read_csv(data_dir / f'fig4A_entropies2{suffix}.csv')

#             fig, ax = plot_scan(
#                 entropies, 
#                 c_field='channel_length',
#                 x_field='interval',
#                 y_field='bitrate_per_hour',
#             )

#             ax.set_ylim(0,1)

#             fig.savefig(panels_dir / f'fig4A{suffix}.svg')
#             fig.savefig(panels_dir / f'fig4A{suffix}.png')
#             plt.close(fig)

fig, ax = subplots_from_axsize(1, 1, (3, 2.5), left=0.7)
 
entropies = pd.read_csv(data_dir / f'fig4A_entropies-c25.csv')

plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    ms=3,
    ax=ax,
)

entropies = pd.read_csv(data_dir / f'fig4A_entropies-cm15.csv')

plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    alpha=0.3,
    ms=3,
    ax=ax,
)

ax.set_ylim(0,1)
handles = ax.get_legend().legend_handles
ax.legend(handles=handles[:len(handles)//2])


fig.savefig(panels_dir / f'fig4A.svg')
fig.savefig(panels_dir / f'fig4A.png')
# plt.close(fig)


channel_lengths = entropies['channel_length'].drop_duplicates().tolist()
print(channel_lengths)

propensities = pd.read_csv(fig3_data_dir / f'propensities.csv').set_index('interval')
probabilities = pd.read_csv(fig3_data_dir / f'probabilities.csv').set_index('interval')
packed_data = pd.read_csv(fig3_data_dir / f'packed_data.csv').set_index(['fate', 'interval'])
expected_fronts = packed_data[['fronts_forward_sum', 'fronts_backward_sum']].div(packed_data['count'], axis=0).fillna(0)
print(expected_fronts)

def expected_number_of_fronts_measured(channel_length, effective_mean_interval):
    # effective_mean_interval = probabilities['fail to start'].index * 2 / (1 - probabilities['fail to start'])
    # print(effective_mean_interval)
    total_event_propensity = propensities['propagation failure'] + propensities['one backward front spawning'] + propensities['other front spawning']
    return (
        (
            expected_fronts.loc['one backward front spawning'].mul(propensities['one backward front spawning'], axis=0)
            + expected_fronts.loc['other front spawning'].mul(propensities['other front spawning'], axis=0)
        ).mul(
            (1 - probabilities['fail to start']) # not failed at start
            * (1 - np.exp(-channel_length * propensities['propagation failure'])) / (channel_length * propensities['propagation failure']).where(propensities['propagation failure'] > 0, 0.) # not failed along the channel
        , axis=0,
        )
    )
    
def expected_number_of_forward_fronts(x, channel_length, effective_mean_interval):
    return np.interp(x, *zip(*expected_number_of_fronts_measured(channel_length, effective_mean_interval)['fronts_forward_sum'].items()))

def expected_number_of_backward_fronts(x, channel_length, effective_mean_interval, v=1/3.6):
    return (channel_length - 0.5 * effective_mean_interval * v) * np.interp(x, *zip(*expected_number_of_fronts_measured(channel_length, effective_mean_interval)['fronts_backward_sum'].items()))


def failure_probability_measured(channel_length):
    return probabilities['fail to start'] + (1 - probabilities['fail to start']) * (1 - np.exp(-channel_length * (propensities['propagation failure'])))
    
def failure_probability(x, channel_length):
    return np.interp(x, *zip(*failure_probability_measured(channel_length).items()))


def total_failure_probability_measured(channel_length):
    return probabilities['fail to start'] + (1 - probabilities['fail to start']) * (1 - np.exp(-channel_length * (propensities['propagation failure'] + propensities['one backward front spawning'] + propensities['other front spawning'])))
    
def total_failure_probability(x, channel_length):
    return np.interp(x, *zip(*total_failure_probability_measured(channel_length).items()))
 

def fail_to_start_probability(x, channel_length):
    return np.interp(x, *zip(*probabilities['fail to start'].items()))


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


def weighted(fn, xx, channel_length, max_gap, **kwargs):
    return sum(2**(-k) * fn(k * xx, channel_length, **kwargs) for k in range(1, max_gap)) + 2**(-max_gap + 1) * fn(max_gap * xx, channel_length, **kwargs)
# fig, ax = subplots_from_axsize(1, 1, (3, 2.5), left=0.7)

xx = np.linspace(20,300,101)
max_gap = 6
for it, channel_length in enumerate(channel_lengths):
    sigma = 1.1  * np.sqrt(channel_length)
    inaccurate_probability = sum((-1)**k * (1-erf((2*k+1) * xx / 2 / (sigma * np.sqrt(2)))) for k in range(5))
    weighted_total_failure_probability = weighted(total_failure_probability, xx, channel_length, max_gap)
    weighted_failure_probability = weighted(failure_probability, xx, channel_length, max_gap)
    effective_mean_interval = xx * 2 / (1 - weighted_total_failure_probability)
    weighted_number_of_forward_fronts = weighted(expected_number_of_forward_fronts, xx, channel_length, max_gap, effective_mean_interval=effective_mean_interval)
    weighted_number_of_backward_fronts = weighted(expected_number_of_backward_fronts, xx, channel_length, max_gap, effective_mean_interval=effective_mean_interval)
    ax.plot(xx, 60 * mi_from_error_probabilities(
        chance_for_missing=weighted_total_failure_probability,
        ) / xx,
        color=f"C{it}")
    ax.plot(xx, 60 * mi_from_error_probabilities(
        chance_for_missing=weighted_total_failure_probability + (1-weighted_total_failure_probability)*inaccurate_probability, 
        chance_for_fake=(1-weighted_total_failure_probability)*inaccurate_probability,
        ) / xx,
        color=f"C{it}", ls=':')
    failure_or_anihilated_probability = weighted_failure_probability + (1 - weighted_failure_probability) * weighted_number_of_backward_fronts
    ax.plot(xx, 60 * mi_from_error_probabilities(
        chance_for_missing=failure_or_anihilated_probability +  (1 - failure_or_anihilated_probability) * inaccurate_probability,
        chance_for_fake=1-np.exp(-weighted_number_of_forward_fronts) + (lambda x: 2*x-x**2)((1 - failure_or_anihilated_probability) * inaccurate_probability),
        ) / xx,
        color=f"C{it}", ls='--')
    # print(expected_number_of_forward_fronts(channel_length))
    # print(expected_number_of_backward_fronts(channel_length))
    # print(expected_number_of_backward_fronts(channel_length))
    print(inaccurate_probability)
    print(pd.DataFrame({
        'weighted_total_failure_probability': weighted_total_failure_probability,
        'weighted_failure_probability': weighted_failure_probability,
        'weighted_number_of_backward_fronts': weighted_number_of_backward_fronts,
        'weighted_number_of_forward_fronts': weighted_number_of_forward_fronts,
        'expected_number_of_backward_fronts': expected_number_of_backward_fronts(xx, channel_length, effective_mean_interval),
        # 'expected_number_of_forward_fronts': expected_number_of_forward_fronts(xx, channel_length, effective_mean_interval),
        },
        index=xx).tail(20))
fig.savefig(panels_dir / 'fig4a-trial-duration1-trained_on_30b.svg')


# shutil.copy2(panels_dir / 'fig4A-c25.svg', panels_dir / 'fig4A.svg')
# shutil.copy2(panels_dir / 'fig4A-c25.png', panels_dir / 'fig4A.png')
(panels_dir / '../../fig5/panels').mkdir(parents=True, exist_ok=True)
shutil.copy2(panels_dir / 'fig4A-c25-reconstruction.svg', panels_dir / '../../fig5/panels/fig5A.svg')
shutil.copy2(panels_dir / 'fig4A-c25-reconstruction.png', panels_dir / '../../fig5/panels/fig5A.png')



for fields in 'c', 'rl':
    for k_neighbors in (15, 25):
        for reconstruction in (True, False):

            suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"

            entropies = pd.read_csv(data_dir / f'fig4A_entropies{suffix}.csv')

            fig, ax = plot_scan(
                entropies, 
                c_field='channel_length',
                x_field='interval',
                y_field='efficiency',
            )

            ax.set_ylabel('efficiency')
            ax.set_ylim(0,1.02)

            fig.savefig(panels_dir / f'figS4-1{suffix}.svg')
            fig.savefig(panels_dir / f'figS4-1{suffix}.png')
            plt.close(fig)


(panels_dir / '../../figS4-1/panels').mkdir(parents=True, exist_ok=True)

shutil.copy2(panels_dir / 'figS4-1-c25.svg', panels_dir / '../../figS4-1/panels/figS4-1.svg')
shutil.copy2(panels_dir / 'figS4-1-c25.png', panels_dir / '../../figS4-1/panels/figS4-1.png')
