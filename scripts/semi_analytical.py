# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import xlogy, erf
from matplotlib import pyplot as plt

import sys
root_repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.analyze_velocity import get_velocity
from scripts.entropy_utils import xlog2x
from scripts.defaults import PARAMETERS_DEFAULT


PREDICTION_TYPE_TO_LS = {
    'failure_and_backward': ':',
    'failure_and_variance': ':',
    'asymptotic_and_variance': ':',
    'failure_backward_and_variance': '--',
    'failure_backward_forward_and_variance': '-',
    'failure_backward_forward_and_variance_v2': '-',
}

channel_lengths = [30,100,300,1000]


fig3_data_dir = lambda channel_length: Path(__file__).parent.parent / 'data' / 'fig3' / 'fig3B' / 'approach1' / f"l-{channel_length}"
fig2_data_dir = Path(__file__).parent.parent / 'data' / 'fig2' / 'fig2C' / 'approach8'
velocity_cache_dir = Path(__file__).parent.parent / 'data' / 'velocity'

avg_n_backward = 1.285
sigma2_0 = get_velocity(channel_width=6, channel_length=300, parameters=PARAMETERS_DEFAULT, velocity_cache_dir=velocity_cache_dir, quantity='variance_per_step') # 1.65

print(sigma2_0)

coefs = pd.read_csv(fig2_data_dir / 'coefs--l-300.csv').set_index('coefficient')

def get_failure_propensity_free_front(channel_width):
    return np.exp(coefs['failure']['a'] * channel_width + coefs['failure']['b'])

def get_spawning_propensity_free_front(channel_width):
    return coefs['spawning']['a'] * channel_width + coefs['spawning']['b']

def get_extinction_probability_distant_fronts(channel_width, channel_length, gamma):
    return 1 - np.exp(-channel_length * (
        get_failure_propensity_free_front(channel_width)
        + gamma * avg_n_backward * get_spawning_propensity_free_front(channel_width)
))

def get_extinction_probability_free_fronts(channel_width, channel_length):
    return 1 - np.exp(-channel_length * get_failure_propensity_free_front(channel_width))

probabilities = pd.concat([pd.read_csv(fig3_data_dir(channel_length) / 'probabilities.csv').set_index('interval') for channel_length in channel_lengths], names=['channel_length'], keys=channel_lengths)
packed_data = pd.concat([pd.read_csv(fig3_data_dir(channel_length) / 'packed_data.csv').set_index(['fate', 'interval']) for channel_length in channel_lengths], names=['channel_length'], keys=channel_lengths)

spawning_probabilities = probabilities['1 backward front spawning'] + probabilities['other front spawning events']
failure_probabilities = probabilities['immediate failure'] + probabilities['propagation failure']

expected_number_of_fronts = packed_data.groupby(['channel_length', 'interval']).sum().fillna(0)
expected_number_of_fronts[['fronts_forward_sum', 'fronts_backward_sum']].div(expected_number_of_fronts['count'], axis=0)  # used only for forward fronts; for backward fronts a constant is used instead
 
def get_for_length(df: pd.Series | pd.DataFrame, channel_length):
    return df[df.index.get_level_values('channel_length') == channel_length].reset_index('channel_length', drop=True)

def interpolate_for_interval(df: pd.Series | pd.DataFrame, interval, channel_length):
    return np.interp(interval, *zip(*get_for_length(df, channel_length).items()))


def get_expected_number_of_backward_fronts(interval, channel_length):
    return avg_n_backward * interpolate_for_interval(spawning_probabilities, interval, channel_length)

def get_failure_probability(interval, channel_length):
    return interpolate_for_interval(failure_probabilities, interval, channel_length)
  
def get_expected_number_of_forward_fronts(interval, channel_length, sigma):
    return (1-erf((x-150) / np.sqrt(2 * channel_length * sigma)))/2 * interpolate_for_intervals(expected_number_of_fronts['fronts_forward_sum'], interval, channel_length) # * (-np.tanh((x - 135) / 30) + 1) / 2


def chance_of_not_hitting_the_start(interval, channel_length, v):
    return (1 - interval * v / (2*channel_length)).clip(0., 1.)


def get_inaccurate_probability(interval, sigma):
    'Computes the probability that |Z| > interval / 2 for Z ~ N(0, sigma**2)'
    return 1 - erf(interval / 2 / (sigma * np.sqrt(2)))


def weighted(fn, interval, channel_length, n_terms=10, sending_probab=0.5, **kwargs):
    'Computes the weighted sum 2**(-k) * fn(k*x)'
    q = sending_probab
    return (
        q / (1-q) * sum((1-q)**k * fn(k * interval, channel_length, **kwargs) for k in range(1, n_terms))
        + (1-q)**(n_terms - 1) * fn(n_terms * interval, channel_length, **kwargs)
        
    )

def compute_mi_from_error_probabilities(sending_probab=.5, chance_for_missing=0., chance_for_fake=0.):
    tp = sending_probab * ((1 - chance_for_missing) + chance_for_missing * chance_for_fake)
    fn = sending_probab * chance_for_missing * (1 - chance_for_fake)
    fp = (1 - sending_probab) * chance_for_fake
    tn = (1 - sending_probab) * (1 - chance_for_fake)
    return (
        (xlog2x(tp) + xlog2x(fn) + xlog2x(fp) + xlog2x(tn)) # -H(S,R)
        - (xlog2x(sending_probab) + xlog2x(1 - sending_probab)) # -H(S)
        - (xlog2x(tp + fp) + xlog2x(tn + fn)) # -H(R)
    )


def get_predictions(intervals, channel_length, sending_probab, prediction_types=[]):

    v = get_velocity(channel_width=6, channel_length=channel_length, parameters=PARAMETERS_DEFAULT, velocity_cache_dir=velocity_cache_dir)

    inaccurate_probability = get_inaccurate_probability(intervals, sigma=np.sqrt(sigma2_0 * channel_length))
    failure_probability = weighted(get_failure_probability, intervals, channel_length, sending_probab=sending_probab)
    # number_of_forward_fronts = weighted(get_expected_number_of_forward_fronts, intervals, channel_length, sigma=sigma, sending_probab=sending_probab)
    number_of_backward_fronts = weighted(get_expected_number_of_backward_fronts, intervals, channel_length, sending_probab=sending_probab)
    # asymptotic_failure_probability = get_failure_probability(300 + 0*intervals, channel_length)
    weighted_gamma = weighted(chance_of_not_hitting_the_start, intervals, channel_length, v=v, sending_probab=sending_probab) # chance that a backward front collides before reaching channel start
    extinction_probability = 1 - (1 - failure_probability) / (1 + number_of_backward_fronts * weighted_gamma) # probability that propagation fails or the front gets annihilated

    mi_per_slot_predictions = {}
    if 'failure_and_backward' in prediction_types:
        mi_per_slot_predictions.update({
            'failure_and_backward': compute_mi_from_error_probabilities(
                sending_probab=sending_probab,
                chance_for_missing=extinction_probability,
                chance_for_fake=0.,
            )
        })
    if 'failure_and_variance' in prediction_types:
        mi_per_slot_predictions.update({
            'failure_and_variance': compute_mi_from_error_probabilities(
                sending_probab=sending_probab,
                chance_for_missing=failure_probability + (1 - failure_probability) * inaccurate_probability,
                chance_for_fake=1-np.exp(-number_of_forward_fronts) + np.exp(-number_of_forward_fronts)*(lambda x: 2*x-x**2)(0.5 * (1 - failure_probability) * inaccurate_probability),
            )
        })
    if 'asymptotic_and_variance' in prediction_types:
        mi_per_slot_predictions.update({
            'asymptotic_and_variance':  compute_mi_from_error_probabilities(
                sending_probab=sending_probab,
                chance_for_missing=asymptotic_extinction_probability + (1 - asymptotic_extinction_probability) * inaccurate_probability,
                chance_for_fake=1-np.exp(-asymptotic_number_of_forward_fronts) + np.exp(-asymptotic_number_of_forward_fronts)*(lambda x: 2*x-x**2)(0.5 * (1 - asymptotic_extinction_probability) * inaccurate_probability),
            )
        })
    if 'failure_backward_and_variance' in prediction_types:
        mi_per_slot_predictions.update({
            'failure_backward_and_variance': compute_mi_from_error_probabilities(
                sending_probab=sending_probab,
                chance_for_missing=extinction_probability + (1 - extinction_probability) * inaccurate_probability,
                chance_for_fake=(lambda x: 2*x-x**2)(0.5 * (1 - extinction_probability) * inaccurate_probability),
            ) 
        })
    if 'backward_forward_and_variance' in prediction_types:
        mi_per_slot_predictions.update({
            'backward_forward_and_variance': compute_mi_from_error_probabilities(
                sending_probab=sending_probab,
                chance_for_missing=extinction_probability + (1 - extinction_probability) * inaccurate_probability,
                chance_for_fake=number_of_forward_fronts / (1 + number_of_backward_fronts * weighted_gamma) + (lambda x: 2*x-x**2)(0.5 * (1 - extinction_probability) * inaccurate_probability),
            )
        })
    if 'backward_forward_and_variance_v2' in prediction_types:
        mi_per_slot_predictions.update({
            'backward_forward_and_variance_v2': compute_mi_from_error_probabilities(
                sending_probab=sending_probab,
                chance_for_missing=extinction_probability + (1 - extinction_probability) * inaccurate_probability,
                chance_for_fake=1-np.exp(-number_of_forward_fronts) + np.exp(-number_of_forward_fronts)*(lambda x: 2*x-x**2)(0.5 * (1 - extinction_probability) * inaccurate_probability),
            )
        })
    
    bit_per_h_predictions = {
        key: val * 60 / intervals 
        for key,val in mi_per_slot_predictions.items()
    }
    return bit_per_h_predictions


def plot_predictions(intervals, channel_length, sending_probab, prediction_types=[], prediction_type_to_ls=PREDICTION_TYPE_TO_LS, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    predictions = get_predictions(intervals, channel_length, sending_probab, prediction_types)

    for prediction_type in prediction_types:
        ax.plot(
            intervals, 
            predictions[prediction_type],
            **(dict(
                color=f"black",
                alpha=0.3,
                ls=prediction_type_to_ls[prediction_type],
                label=prediction_type,
            ) | kwargs)
            )

    ax.set_xlabel('interval [min]')
    ax.set_ylabel('bitrate [bit/hour]')

