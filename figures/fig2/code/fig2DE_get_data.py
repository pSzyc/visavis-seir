import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / 'fig2C' / 'approach8'

channel_widths = list(range(1,10)) + list(range(10,21,2))
channel_length = 300
n_simulations = 30000

field_forward = 'forward'
field_backward = 'backward'

propensities = pd.read_csv(data_dir / 'fig2C--propensities.csv').set_index(['channel_width', 'channel_length'])

events_parts = {}
for w in channel_widths:
    first_split_events = pd.read_csv(data_dir / f'w-{w}-l-{channel_length}' / 'first_split_events.csv').set_index(['channel_width', 'channel_length', 'simulation_id'])

    counts = (
        first_split_events[[field_forward, field_backward]]
            .reindex(list(product([w], [channel_length], range(n_simulations))), fill_value=0)
            .value_counts([field_forward, field_backward])
            .reset_index()
            .assign(
                total_spawned=lambda df: df[field_forward] + df[field_backward],
                channel_width=w,
            )
            .rename(columns={0: 'count'})
    )
    counts_spawning = counts[counts[field_forward].gt(0) | counts[field_backward].gt(0)].copy()
    counts_spawning['propensity'] = propensities.loc[w, channel_length]['l_spawning'] * counts_spawning['count'] / counts_spawning['count'].sum()
    events_parts.update({w: counts_spawning})

events = pd.concat(events_parts, ignore_index=True).set_index('channel_width')
events.to_csv(data_dir / 'event_counts.csv')


expected_front_count = pd.DataFrame({
    'forward':  (events[field_forward] * events['count']).groupby('channel_width').sum() / events['count'].groupby('channel_width').sum(),
    'backward': (events[field_backward] * events['count']).groupby('channel_width').sum() / events['count'].groupby('channel_width').sum(),
})

expected_front_count.to_csv(data_dir / 'expected_front_count.csv')
print(expected_front_count)


specific = pd.DataFrame({
        '1 backward':
            events[events[field_forward].eq(0) & events[field_backward].eq(1)].groupby('channel_width')['propensity'].sum(),
        '1 forward, 1 backward':
            events[events[field_forward].eq(1) & events[field_backward].eq(1)].groupby('channel_width')['propensity'].sum(),
        '1 forward':
            events[events[field_forward].eq(1) & events[field_backward].eq(0)].groupby('channel_width')['propensity'].sum(),
        'other events with $\\leq$ 6 spawned fronts':
            events[events['total_spawned'].le(6) & (events[field_forward].gt(1) | events[field_backward].gt(1))].groupby('channel_width')['propensity'].sum(),
        'events with > 6 spawned fronts':
            events[events['total_spawned'].gt(6)].groupby('channel_width')['propensity'].sum(),
    })


specific.to_csv(data_dir / 'detailed_event_propensities.csv')


