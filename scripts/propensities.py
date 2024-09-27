# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
import numpy as np
from pathlib import Path
import pandas as pd
from itertools import product

from .analyze_tracking import generate_dataset
from .utils import simple_starmap

# def count_events(df):
#     df_results = df.loc[:, ['failure', 'spawning']]
#     df_results['first_event_position'] = df_results[['failure', 'spawning']].min(axis=1)
#     df_results['first_event'] = df_results.fillna(np.inf).idxmin(axis=1).replace(np.inf, np.nan)
#     mean_steps = df_results['first_event_position'].mean()
#     events = df_results['first_event'].value_counts().reindex(['failure', 'spawning']).fillna(0).astype(int)
#     return events, mean_steps

def count_events(pulse_fates):
    first_event_position = pulse_fates[['failure', 'spawning']].min(axis=1)
    first_event = pulse_fates.fillna(np.inf).idxmin(axis=1).mask(first_event_position.isna())
    mean_steps = first_event_position.mean()
    events = first_event.value_counts().reindex(['failure', 'spawning']).fillna(0).astype(int)
    return events, mean_steps

def compute_propensities(n_any_event, mean_steps, n_samples, length, event_counts):
    l_tot = n_any_event / (n_any_event * mean_steps + (n_samples - n_any_event) * length)
    l_spawning = event_counts['spawning'] / n_any_event * l_tot
    l_failure = event_counts['failure'] / n_any_event * l_tot
    return l_tot, l_spawning, l_failure

def get_propensities(channel_length, channel_width, n_simulations, outdir=None, **kwargs):
    print('----------', 'W =', channel_width)
    pulse_fates = generate_dataset(
        input_protocol=[],
        n_simulations=n_simulations,
        channel_width=channel_width,
        channel_length=channel_length,
        outdir=outdir,
        **kwargs,
        )

    # Extintion position if failure
    pulse_fates['failure'] = (
        pulse_fates['track_end_position']
            .fillna(1 if channel_width < 7 else np.nan) # Fill with 1 if lost_somewhere (typical cause is initiation failure for W < 7)
            .where(pulse_fates['fate'] == 'lost_somewhere', 
                pulse_fates['track_end_position'].where(
                    pulse_fates['fate'] == 'failure',
                    np.nan
        ))) - 1 # minus one, as the simulation starts from h=1
    
    # First pulse spawning position if spawning
    pulse_fates['spawning'] = pulse_fates['significant_split_position'] - 1# minus one, as the simulation starts from h=1


    event_counts, mean_steps = count_events(pulse_fates[['failure', 'spawning']])
    any_event_count = event_counts.sum()
    l_tot, l_spawning, l_failure = compute_propensities(any_event_count, mean_steps, n_simulations, channel_length - 1, event_counts) # minus one, as the simulation starts from h=1
    return {
        'channel_width': channel_width,
        'channel_length': channel_length,
        'l': l_tot,
        'l_spawning': l_spawning,
        'l_failure': l_failure,
    }


def get_propensities_batch(channel_lengths, channel_widths, n_simulations, results_file, n_workers, use_cached=False, per_width_kwargs={}, per_length_kwargs={}, **kwargs):

    if use_cached and results_file.exists():
        propensities = pd.read_csv(results_file).set_index(['channel_width', 'channel_length'])
        if propensities.index.unique().tolist() == sorted(product(channel_widths, channel_lengths)):
            return propensities

    propensities = pd.DataFrame(simple_starmap(
        get_propensities,
        [
            dict(
                channel_length=channel_length,
                channel_width=channel_width,
                
                n_simulations=n_simulations,
                outdir=(results_file.parent / f'w-{channel_width}-l-{channel_length}'),
                n_workers=n_workers,
                use_cached=use_cached,
                ) 
                | kwargs 
                | (per_width_kwargs[channel_width] if channel_width in per_width_kwargs else {}) 
                | (per_length_kwargs[channel_length] if channel_length in per_length_kwargs else {})
            for channel_width in channel_widths for channel_length in channel_lengths
        ],
        )).set_index(['channel_width', 'channel_length'])
    propensities.to_csv(results_file)
    return propensities

