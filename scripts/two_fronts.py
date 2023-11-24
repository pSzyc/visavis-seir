import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from scripts.defaults import PARAMETERS_DEFAULT
from scripts.analyze_tracking import generate_dataset 
from scripts.utils import simple_starmap
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
        n_margin=0,
        outdir=None,
        plot_results=False,
        save_states=True,
        save_iterations=True,
        use_cached=False,
        **kwargs,
        ):

    front_fields = (
        ['forward', 'backward'] if fate_criterion == 'generated' else 
        ['reached_end', 'reached_start'] if fate_criterion == 'reached' else
        []
    )

    if use_cached and outdir:
        data = pd.read_csv(outdir / 'pulse_fates.csv').set_index(['channel_length', 'channel_width', 'simulation_id', 'Unnamed: 0'])
    else:
        data = generate_dataset(
            input_protocol=[interval],
            n_simulations=n_simulations,
            parameters=parameters,
            channel_width=channel_width,
            channel_length=channel_length,
            duration=duration,
            interval_after=interval_after,
            n_margin=n_margin,
            outdir=outdir,
            plot_results=plot_results,
            save_states=save_states,
            save_iterations=save_iterations,
            **kwargs
            )
    data = data[data['pulse_id'] == 1]
    data['interval'] = interval
    data['is_spawning'] = (
        1*(data['reached_start'].gt(0) | data['reached_end'].gt(1)) + 1*(data['reached_start'].gt(1) | data['reached_end'].gt(1)) if fate_criterion == 'reached' 
        else 1*(data['backward'].gt(0) | data['forward'].gt(0)) + 1 * (data['backward'].gt(1) | data['forward'].gt(0)) if fate_criterion == 'generated'
        else np.nan
    )
    data['first_event_position'] = data[['significant_split_position', 'track_end_position']].min(axis=1).fillna(0)

    counts = data.value_counts(['channel_length', 'channel_width', 'interval', 'fate', 'is_spawning']).sort_index() # add front_fields if necessary
    data_sum = data.groupby(['channel_length', 'channel_width', 'interval', 'fate', 'is_spawning']).sum() # add front_fields if necessary
    
    pulse_fate_count = pd.DataFrame({
        'count': counts,
        'first_event_position_mean':  data_sum['first_event_position'] / counts,
        'first_event_position_sum':  data_sum['first_event_position'],
        'fronts_forward_sum':  data_sum[front_fields[0]] - (counts if fate_criterion == 'reached' else 0) * (data_sum.index.get_level_values('fate') == 'transmitted'),
        'fronts_backward_sum':  data_sum[front_fields[1]],
        })
    if outdir:
        pulse_fate_count.to_csv(outdir / 'pulse_fate_count.csv')
    return pulse_fate_count



def get_propensities(interval, n_sim, channel_width, outdir=None, channel_length=300, **kwargs):
    print('----------', channel_width)
    pulse_fates = get_pulse_fate_counts(interval, n_simulations=n_sim, channel_width=channel_width, channel_length=channel_length, outdir=outdir, save_states=False, **kwargs)
    return pulse_fates



def simulate(n_sim, channel_widths, intervals, results_file, channel_length, n_workers, per_width_kwargs={}, **kwargs):
    counts = simple_starmap(
        get_propensities,
        [
            dict(
                channel_length=channel_length,
                n_sim=n_sim,
                interval=interval,
                channel_width=w,
                outdir=(results_file.parent / f'w-{w}-l-{channel_length}' / f'interval-{interval}'),
                n_workers=n_workers,
                ) | kwargs | (per_width_kwargs[w] if w in per_width_kwargs else {})
            for w in channel_widths for interval in intervals
        ],
        )
    
    data = pd.concat(counts)
    data.to_csv(results_file)
    return data
if __name__ == '__main__':
    simulate()