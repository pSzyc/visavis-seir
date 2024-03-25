import numpy as np
import csv
import click
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_tracking import generate_dataset
from scripts.utils import simple_starmap

def get_results(df):
    df_results = df.loc[:, ['failure', 'spawning']]
    df_results['first_event_position'] = df_results[['failure', 'spawning']].min(axis=1)
    df_results['first_event'] = df_results.idxmin(axis=1)
    mean_steps = df_results['first_event_position'].mean()
    events = df_results['first_event'].value_counts().reindex(['failure', 'spawning']).fillna(0).astype(int)
    return events, mean_stepsd

def compute_propensities(n_any_event, mean_steps, n_samples, length, event_counts):
    l_tot = n_any_event / (n_any_event * mean_steps + (n_samples - n_any_event) * length)
    l_spawning = event_counts['spawning'] / n_any_event * l_tot
    l_failure = event_counts['failure'] / n_any_event * l_tot
    return l_tot, l_spawning, l_failure

def get_propensities(channel_length, n_sim, channel_width, outdir=None, **kwargs):
    print('----------', channel_width)
    pulse_fates = generate_dataset([], n_simulations=n_sim, channel_width=channel_width, channel_length=channel_length, outdir=outdir, save_states=False, **kwargs)

    # Extintion position if failure
    pulse_fates['failure'] = (
        pulse_fates['track_end_position']
            .fillna(1 if channel_width < 7 else np.nan) # Data for W >= 7 too noisy for fitting
            .where(pulse_fates['fate'] == 'lost_somewhere', 
                pulse_fates['track_end_position'].where(
                    pulse_fates['fate'] == 'failure',
                    np.nan
        ))) - 1 # minus one, as the simulation starts from h=1
    
    # First pulse spawning position if spawning
    pulse_fates['spawning'] = pulse_fates['significant_split_position'] - 1# minus one, as the simulation starts from h=1

    event_counts, mean_steps = get_results(pulse_fates[['failure', 'spawning']])
    any_event_count = event_counts.sum()
    l_tot, l_spawning, l_failure = compute_propensities(any_event_count, mean_steps, n_sim, channel_length - 1, event_counts) # minus one, as the simulation starts from h=1
    return {
        'channel_width': channel_width,
        'channel_length': channel_length,
        'l': l_tot,
        'l_spawning': l_spawning,
        'l_failure': l_failure,
    }

# @click.command()
# @click.argument('n_sim', type=int)
# @click.argument('channel_widths', type=int)
# @click.argument('results_file', type=str, default='lambda.csv')
# @click.argument('channel_length', type=int, default=300)
# @click.argument('n_workers', type=int, default=5)
def simulate(n_sim, channel_widths, results_file, channel_length, n_workers, per_width_kwargs={}, **kwargs):

    # with Pool(n_workers) as pool:
    propensities = pd.DataFrame(simple_starmap(
        get_propensities,
        [
            dict(
                channel_length=channel_length,
                n_sim=n_sim,
                channel_width=w,
                outdir=(results_file.parent / f'w-{w}-l-{channel_length}'),
                n_workers=n_workers,
                
                ) | kwargs | (per_width_kwargs[w] if w in per_width_kwargs else {})
            for w in channel_widths
        ],
        )).set_index(['channel_width', 'channel_length'])
    propensities.to_csv(results_file)
    return propensities

if __name__ == '__main__':
    simulate()