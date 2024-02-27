import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.significant_splits import get_effective_front_directions, get_hierarchy, get_split_events
from scripts.utils import starmap

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / 'fig2C' / 'approach8'

channel_widths = list(range(1,10)) + list(range(10,21,2))
channel_length = 300
n_simulations = 30000


def extract_single(channel_length, channel_width, simulation_id):
    outdir = data_dir / f'w-{channel_width}-l-{channel_length}' / f'sim-{simulation_id}'

    track_info = pd.read_csv(outdir / 'track_info.csv').set_index('track_id')
    significant_splits = pd.read_csv(outdir / 'significant_splits.csv').set_index('track_id')

    effective_front_directions = get_effective_front_directions(track_info)
    hierarchy = get_hierarchy(significant_splits, track_info)
    significant_splits_with_hierarchy = significant_splits.join(hierarchy)
    split_events = get_split_events(significant_splits_with_hierarchy, effective_front_directions, track_info)

    effective_front_directions.to_csv(outdir / 'effevtive_front_directions.csv')
    hierarchy.to_csv(outdir / 'significant_split_hierarchy.csv')
    split_events.to_csv(outdir / 'split_events.csv')

    return split_events

for channel_width in channel_widths:

    outdir = data_dir / f'w-{channel_width}-l-{channel_length}'

    split_events = pd.concat(list(starmap(
        extract_single,
        [dict(
            channel_length=channel_length,
            channel_width=channel_width,
            simulation_id=simulation_id,
        )
        for channel_length, channel_width, simulation_id in product([channel_length], [channel_width], range(n_simulations))
        ],
        processes=5,
    )),
        names=['channel_length', 'channel_width', 'simulation_id'],
        keys=list(product([channel_length], [channel_width], range(n_simulations))),
    )

    first_split_events = split_events[split_events.index.get_level_values('event_id') == 0]
    first_split_events.to_csv(outdir / 'first_split_events.csv')
    print(channel_width)






