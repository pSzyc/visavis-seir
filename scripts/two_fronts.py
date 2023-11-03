import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numba as np
from scripts.analyze_tracking import generate_dataset
from scripts.propensities import get_results, compute_propensities
from scripts.utils import starmap
import csv

def get_propensities(interval, n_sim, channel_width, outdir=None, channel_length=300, **kwargs):
    print('----------', channel_width)
    pulse_fates = generate_dataset([interval], n_simulations=n_sim, channel_width=channel_width, channel_length=channel_length, outdir=outdir, save_states=False, **kwargs)
    pulse_fates = pulse_fates[pulse_fates['pulse_id'] == 1]

    # Extintion position if failure
    pulse_fates['failure'] = (
        (pulse_fates['track_end_position'].fillna(1 if channel_width < 7 else np.nan)).where(pulse_fates['fate'] == 'lost_somewhere',
        pulse_fates['track_end_position'].where(pulse_fates['fate'] == 'failure',
        np.nan
        ))) - 1 # minus one, as the simulation starts from h=1
    
    # Spawning position if spawning
    pulse_fates['spawning'] = pulse_fates['significant_split_position'] - 1# minus one, as the simulation starts from h=1
    event_counts, mean_steps = get_results(pulse_fates[['failure', 'spawning']])
    any_event_count = event_counts.sum()
    l, l_chaos, l_ext = compute_propensities(any_event_count, mean_steps, n_sim, channel_length - 1, event_counts) # minus one, as the simulation starts from h=1
    return channel_width, channel_length, interval, l, l_chaos, l_ext

def to_csv(results, result_file):
    with open(result_file, 'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['channel_width', 'channel_length', 'interval', 'l', 'l_spawning', 'l_failure'])
        for row in results:
            csv_out.writerow(row)

def simulate(n_sim, channel_widths, intervals, results_file, channel_length, n_workers, per_width_kwargs={}, **kwargs):
    propensities = starmap(
        get_propensities,
        [
            dict(
                channel_length=channel_length,
                n_sim=n_sim,
                interval=interval,
                channel_width=w,
                outdir=(results_file.parent / f'w-{w}-l-{channel_length}'/ f'interval-{interval}'),
                
                ) | kwargs | (per_width_kwargs[w] if w in per_width_kwargs else {})
            for w in channel_widths for interval in intervals
        ],
        processes=n_workers,
        )
    to_csv(propensities, results_file)

if __name__ == '__main__':
    simulate()