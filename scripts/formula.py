import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import xlogy, erf

import sys
root_repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.defaults import PARAMETERS_DEFAULT
from scripts.entropy_utils import get_cycle_time_std, get_cycle_time

LOG2 = np.log(2)

def xlogx(x):
    return xlogy(x,x) / LOG2

def gaussian_cdf(x):
    return (erf(x/np.sqrt(2)) + 1) / 2


figS1_1_data_dir = Path(__file__).parent.parent / 'data' / 'figS1' / 'approach5'
velocities = pd.read_csv(figS1_1_data_dir / 'velocities.csv')

figS1_3_data_dir = Path(__file__).parent.parent / 'data' / 'figS1' / 'figS1-3' / "approach1"
variance_data = pd.read_csv(figS1_3_data_dir / 'variance_per_step.csv')

fig2C_data_dir = Path(__file__).parent.parent / 'data' / 'fig2' / 'fig2C' / 'approach8'
coefs = pd.read_csv(fig2C_data_dir / 'coefs.csv').set_index('Unnamed: 0')

fig2E_data_dir = Path(__file__).parent.parent / 'data' / 'fig2' / 'fig2C' / 'approach8'
specific = pd.read_csv(fig2C_data_dir / 'specific.csv').set_index('channel_width').fillna(0)
permanent_share = specific['total with > 6 spawned fronts'] / specific.sum(axis=1)



def get_total_lambda(channel_width):
    lambda_sp = coefs['spawning']['a'] * channel_width + coefs['spawning']['b']
    lambda_fail = np.exp(coefs['failure']['a'] * channel_width + coefs['failure']['b'])
    return lambda_sp + lambda_fail

def get_permament_spawning_site_lambda(channel_width):
    lambda_sp = coefs['spawning']['a'] * channel_width + coefs['spawning']['b']
    share = np.interp(channel_width, permanent_share.index, permanent_share)
    return lambda_sp * share


def get_time_var(channel_width):
    return np.interp(channel_width, variance_data['channel_width'], variance_data['variance_per_step'])

def get_velocity(channel_width):
    # return np.interp(channel_width, velocities['channel_width'], velocities['velocity'])
    return channel_width * 0 + 1/3.6

def predicted_optimal_interval_formula(channel_width, channel_length, parameters=PARAMETERS_DEFAULT):
    return (get_cycle_time(parameters) + get_cycle_time_std(parameters) + np.e/1.5 * np.sqrt(get_time_var(channel_width) * channel_length))
    return np.sqrt((get_cycle_time(parameters) + get_cycle_time_std(parameters))**2 + (np.e * np.sqrt(get_time_var(channel_width) * channel_length))**2)
    return (
        (get_cycle_time(parameters)# - 15
        + (np.log(channel_length) - .3)/3 * get_cycle_time_std(parameters) # np.log(np.sqrt(channel_length)/2)
        + (np.log(channel_length) - .3)/3/2 * np.sqrt(get_time_var(channel_width) * channel_length)) #* get_cycle_time_std(parameters)
    )


def predicted_bitrate_formula(channel_width, channel_length, parameters=PARAMETERS_DEFAULT):
    return (
        max(get_cycle_time(parameters) 
        + np.log(np.sqrt(channel_length)/2) * get_cycle_time_std(parameters),
        np.e * np.sqrt(get_time_var(channel_width) * channel_length))
    )
    return (
        max(get_cycle_time(parameters) 
        + np.log(np.sqrt(channel_length)/2) * get_cycle_time_std(parameters),
        np.e * np.sqrt(get_time_var(channel_width) * channel_length))
    )

def get_factors(channel_width, channel_length, parameters=PARAMETERS_DEFAULT):
    # range_fail_probab = 1 - np.exp(-channel_length * get_total_lambda(channel_width))
    # factor_range = - (1 + range_fail_probab) / 2 * (xlogx(1 / (1 + range_fail_probab)) + xlogx(range_fail_probab / (1 + range_fail_probab)))
    
    # opt_interval = get_cycle_time(parameters) + .5* get_cycle_time_std(parameters) + np.e * np.sqrt(get_time_var(channel_width) * channel_length)
    # std = np.sqrt(get_time_var(channel_width) * channel_length)
    # factor_variance = 1-(2*gaussian_cdf(opt_interval / std / 2) - 1)**2#(1 - 0.25 * np.e) + 0 * (channel_width + channel_length)
    # factor_refractory = (
    #     (1 - 0.25 * np.e) 
    #     * (get_cycle_time(parameters + np.log(np.sqrt(channel_length)/2) * get_cycle_time_std(parameters)) 
    #     / (np.e * np.sqrt(get_time_var(channel_width) * channel_length))
    # )


    temporal_scale = np.e * np.sqrt(get_time_var(channel_width) * channel_length)
    refractory_time = get_cycle_time(parameters)

    factor_range = channel_length * get_total_lambda(channel_width)
    factor_variance = temporal_scale**12 / (temporal_scale**12 + refractory_time**12) #/ get_velocity(channel_width)
    factor_refractory = refractory_time**12 / (temporal_scale**12 + refractory_time**12) #/ get_velocity(channel_width)
    factors = np.array([factor_range, factor_variance, factor_refractory])
    return factors / factors.sum(axis=0), get_velocity(channel_width)



def get_range(channel_width):
    return 1 / get_total_lambda(channel_width)

def get_length_with_variance_equal_refractory(channel_width, parameters=PARAMETERS_DEFAULT):
    return (get_cycle_time(parameters) / np.e)**2 / get_time_var(channel_width)
