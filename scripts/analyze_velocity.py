from pathlib import Path
import pandas as pd
from warnings import warn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) # in order to be able to import from scripts.py

from scripts.analyze_periodic import generate_dataset_batch
from scripts.formula import get_velocity_formula


def get_velocities(
    outdir=None, 
    return_variance=False, 
    **kwargs
    ):

    arrival_times = generate_dataset_batch(
        n_pulses=0,
        intervals=[0],
        outdir=outdir,
        **kwargs
        )

    first_arrival_times = arrival_times.groupby(['channel_width', 'channel_length', 'simulation_id'])['seconds'].min()
    mean_transit_times = first_arrival_times.groupby(['channel_width', 'channel_length']).mean()
    velocity = 1 / mean_transit_times * (mean_transit_times.index.get_level_values('channel_length') - 2)

    velocity.name = 'velocity'
    if outdir:
        velocity.to_csv(outdir  / 'velocity.csv')


    if return_variance:
        transit_time_variance = first_arrival_times.groupby(['channel_width', 'channel_length']).var()
        variance_per_step = transit_time_variance / (transit_time_variance.index.get_level_values('channel_length') - 2)
        variance_per_step.name = 'variance_per_step'
        if outdir:
            variance_per_step.to_csv(data_dir  / 'variance_per_step.csv')
        return velocity, variance_per_step

    return velocity


def get_velocity(channel_width, channel_length, parameters, velocity_cache_dir, duration=5, processes=20):
    
    cache_dir = (
        velocity_cache_dir / 
        'c_rate' / f"{parameters['c_rate']:.3f}" 
        / 'e_forward_rate' / f"{parameters['e_forward_rate']:.3f}" 
        / 'e_subcompartments_count' / f"{parameters['e_subcompartments_count']:d}" 
        / 'i_forward_rate' / f"{parameters['i_forward_rate']:.3f}" 
        / 'i_subcompartments_count' / f"{parameters['i_subcompartments_count']:d}" 
        / f"l-{channel_length}--w-{channel_width}"
        )

    estimated_velocity = get_velocity_formula(parameters)

    if (cache_dir / 'velocity.csv').exists():
        velocity = pd.read_csv(cache_dir / 'velocity.csv').set_index(['channel_width', 'channel_length'])['velocity']
    else:
        velocity = get_velocities(
            channel_widths=[channel_width],
            channel_lengths=[channel_length],
            parameters=parameters,
            n_simulations=3000,
            duration=duration,
            interval_after=int((2 * channel_length/estimated_velocity) // duration) * duration + 200,
            processes=processes,
        )
        cache_dir.mkdir(exist_ok=True, parents=True)
        velocity.to_csv(cache_dir / 'velocity.csv')

    if len(velocity):
        return velocity.loc[channel_width, channel_length]
    warn("No fronts reached channel end during velocity estimation. Using analytical prediction.")
    return estimated_velocity

    





