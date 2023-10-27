import numpy as np
import csv
import click
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_tracking import generate_dataset

def get_results(df):
    df_results = df.loc[:, ['failure', 'spawning']]
    df_results['first_event_position'] = df_results[['failure', 'spawning']].min(axis=1)
    df_results['first_event'] = df_results.idxmin(axis=1)
    mean_steps = df_results['first_event_position'].mean()
    events = df_results['first_event'].value_counts().reindex(['failure', 'spawning']).fillna(0).astype(int)
    return events, mean_steps

def compute_propensities(n_any_event, mean_steps, n_samples, length, event_counts):
    l = n_any_event / (n_any_event * mean_steps + (n_samples - n_any_event) * length)
    l_spawning = event_counts['spawning'] / n_any_event * l
    l_failure = event_counts['failure'] / n_any_event * l
    return l, l_spawning, l_failure

def get_propensities(length, n_sim, width, outdir=None):
    print('----------', width)
    pulse_fates = generate_dataset([], n_simulations=n_sim, channel_width=width, channel_length=length, outdir=outdir, save_states=False) #pd.read_csv(outdir / 'pulse_fates.csv')

    # Extintion position if failure
    pulse_fates['failure'] = pulse_fates['track_end_position'].where(pulse_fates['fate'] == 'failure', np.nan)
    # First pulse spawning position if spawning
    pulse_fates['spawning'] = pulse_fates['significant_split_position']

    event_counts, mean_steps = get_results(pulse_fates)
    any_event_count = event_counts.sum()
    l, l_chaos, l_ext = compute_propensities(any_event_count, mean_steps, n_sim, length, event_counts)
    return width, length, l, l_chaos, l_ext

def to_csv(results, result_file):
    with open(result_file, 'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['channel_width', 'channel_length', 'l', 'l_spawning', 'l_failure'])
        for row in results:
            csv_out.writerow(row)

# @click.command()
# @click.argument('n_sim', type=int)
# @click.argument('channel_widths', type=int)
# @click.argument('results_file', type=str, default='lambda.csv')
# @click.argument('channel_length', type=int, default=300)
# @click.argument('n_workers', type=int, default=5)
def simulate(n_sim, channel_widths, results_file, channel_length, n_workers):
    results = []

    with Pool(n_workers) as pool:
            results = pool.starmap(get_propensities, [(channel_length, n_sim, w, (results_file.parent / f'w{w}-l-{channel_length}'))  for w in channel_widths])
    print(results)
    to_csv(results, results_file)

if __name__ == '__main__':
    simulate()