import numpy as np
import pandas as pd
from .defaults import PARAMETERS_DEFAULT
from .analyze_tracking import generate_dataset
from .utils import simple_starmap

def get_pulse_fate_counts(
        interval,
        channel_width=6,
        channel_length=300,
        fate_criterion='reached',
        outdir=None,
        use_cached=False,
        **kwargs,
        ):

    front_fields = (
        ['forward', 'backward'] if fate_criterion == 'generated' else 
        ['reached_end', 'reached_start'] if fate_criterion == 'reached' else
        []
    )

    if use_cached and outdir:
        data = pd.read_csv(outdir / 'pulse_fates.csv').set_index(['channel_length', 'channel_width', 'simulation_id'])
    else:
        data = generate_dataset(
            input_protocol=[interval],
            channel_width=channel_width,
            channel_length=channel_length,
            outdir=outdir,
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



def get_pulse_fate_counts_batch(channel_widths, channel_lengths, intervals, n_simulations, results_file, n_workers, per_width_kwargs={}, per_length_kwargs={}, **kwargs):
    data = pd.concat(list(simple_starmap(
        get_pulse_fate_counts,
        [
            dict(
                channel_width=channel_width,
                channel_length=channel_length,
                interval=interval,
                n_simulations=n_simulations,
                outdir=(results_file.parent / f'w-{channel_width}-l-{channel_length}' / f'interval-{interval}'),
                n_workers=n_workers,
                ) 
                | kwargs 
                | (per_width_kwargs[channel_width] if channel_width in per_width_kwargs else {})
                | (per_length_kwargs[channel_length] if channel_length in per_length_kwargs else {})
            for channel_width in channel_widths for channel_length in channel_lengths for interval in intervals
        ],
        )))
    
    data.to_csv(results_file)
    return data
