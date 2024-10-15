# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
from pathlib import Path
import pandas as pd
from warnings import warn
from typing import Literal

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
            variance_per_step.to_csv(outdir / 'variance_per_step.csv')
        return velocity, variance_per_step

    return velocity


def get_velocity(channel_width, channel_length, parameters, velocity_cache_dir, quantity: Literal['velocity'] | Literal['variance_per_step'] = 'velocity', logging_interval=5, use_cached=True, processes=20):
    
    cache_dir = (
        velocity_cache_dir / 
        'c_rate' '-' f"{parameters['c_rate']:.3f}" 
        '--' 'e_forward_rate' '-' f"{parameters['e_forward_rate']:.3f}" 
        '--' 'e_subcompartments_count' '-' f"{parameters['e_subcompartments_count']:d}" 
        '--' 'i_forward_rate' '-' f"{parameters['i_forward_rate']:.3f}" 
        '--' 'i_subcompartments_count' '-' f"{parameters['i_subcompartments_count']:d}" 
        '/' f"l-{channel_length}--w-{channel_width}"
        )

    estimated_velocity = get_velocity_formula(parameters)

    if use_cached and (cache_dir / f'{quantity}.csv').exists():
        velocity = pd.read_csv(cache_dir / f'{quantity}.csv').set_index(['channel_width', 'channel_length'])[quantity]
    else:
        velocity = get_velocities(
            channel_widths=[channel_width],
            channel_lengths=[channel_length],
            parameters=parameters,
            n_simulations=3000,
            logging_interval=logging_interval,
            interval_after=int((2 * channel_length/estimated_velocity) // logging_interval) * logging_interval + 200,
            processes=processes,
            return_variance = quantity == 'variance_per_step',
        )
        if quantity == 'variance_per_step':
            velocity = velocity[1]
        cache_dir.mkdir(exist_ok=True, parents=True)
        velocity.to_csv(cache_dir / f'{quantity}.csv')

    if len(velocity):
        return velocity.loc[channel_width, channel_length]
    warn("No fronts reached channel end during velocity estimation. Using analytical prediction.")
    return estimated_velocity

    






