import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from scripts.defaults import PARAMETERS_DEFAULT
from scripts.analyze_tracking import generate_dataset 
from scripts.utils import starmap
import pandas as pd

def get_pulse_fate_counts(
        interval,
        n_simulations,
        parameters=PARAMETERS_DEFAULT,
        channel_width=7,
        channel_length=300,
        duration=5,
        interval_after=1500,
        fate_criterion='reached',
        offset=None,
        n_margin=0,
        outdir=None,
        plot_results=False,
        save_states=True,
        save_iterations=True,
        **kwargs,
        ):

    fields_to_groupby = (
        ['fate', 'forward', 'backward'] if fate_criterion == 'generated' else 
        ['fate', 'reached_end', 'reached_start'] if fate_criterion == 'reached' else
        []
    )
    data = generate_dataset(
        input_protocol=[interval],
        n_simulations=n_simulations,
        parameters=parameters,
        channel_width=channel_width,
        channel_length=channel_length,
        duration=duration,
        interval_after=interval_after,
        offset=offset,
        n_margin=n_margin,
        outdir=outdir,
        plot_results=plot_results,
        save_states=save_states,
        save_iterations=save_iterations,
        **kwargs
        )
    data = data[data['pulse_id'] == 1]
    data['interval'] = interval
    counts = data.value_counts(['channel_length', 'channel_width', 'interval'] + fields_to_groupby).sort_index()
    
    counts = pd.DataFrame({'count': counts})
    if outdir:
        counts.to_csv(outdir / 'pulse_fate_count.csv')
    return counts



def get_propensities(interval, n_sim, channel_width, outdir=None, channel_length=300, **kwargs):
    print('----------', channel_width)
    pulse_fates = get_pulse_fate_counts(interval, n_simulations=n_sim, channel_width=channel_width, channel_length=channel_length, outdir=outdir, save_states=False, **kwargs)
    return pulse_fates



def simulate(n_sim, channel_widths, intervals, results_file, channel_length, n_workers, per_width_kwargs={}, **kwargs):
    counts = starmap(
        get_propensities,
        [
            dict(
                channel_length=channel_length,
                n_sim=n_sim,
                interval=interval,
                channel_width=w,                
                ) | kwargs | (per_width_kwargs[w] if w in per_width_kwargs else {})
            for w in channel_widths for interval in intervals
        ],
        processes=n_workers,
        )
    
    data = pd.concat(counts)
    data.to_csv(results_file)
if __name__ == '__main__':
    simulate()