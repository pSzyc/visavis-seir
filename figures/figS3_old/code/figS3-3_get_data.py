from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3'/ 'figS3-3' / 'approach4'

from scripts.analyze_tracking import generate_dataset
from scripts.utils import simple_starmap


def simulate(n_sim, channel_widths, results_file, channel_lengths, per_width_kwargs={}, per_length_kwargs={}, **kwargs):

    # with Pool(n_workers) as pool:
    pulse_fates = pd.concat(simple_starmap(
        generate_dataset,
        [
            dict(
                input_protocol=[],
                channel_length=l,
                n_simulations=n_sim,
                channel_width=w,
                outdir=(results_file.parent / f'w-{w}-l-{l}'),
                save_states=False,
                ) | kwargs | (per_width_kwargs[w] if w in per_width_kwargs else {}) | (per_length_kwargs[l] if l in per_length_kwargs else {})
            for w, l in product(channel_widths, channel_lengths)
        ]
        ))#.set_index(['channel_width', 'channel_length', 'simulation_id', 'pulse_id'])
    pulse_fates.to_csv(results_file)
    return pulse_fates


# channel_lengths = [30,100,300,520,1000]
channel_lengths = [300]
channel_widths = list(range(1,10)) + list(range(10,21,2))


simulate(
    n_sim=3000,
    channel_widths=channel_widths,
    channel_lengths=channel_lengths,
    results_file= data_dir / 'pulse_fates.csv',
    n_workers=20,
    # use_cached=True,
    per_width_kwargs = {
        w: {
            'front_direction_minimal_distance': min(max(w - 1, 1), 5),
            'min_peak_height': 0.03 / w,
        } for w in channel_widths
    },
    per_length_kwargs={
        l: {
        'interval_after': int(l * 4.2),
        }
        for l in channel_lengths
    }
)



