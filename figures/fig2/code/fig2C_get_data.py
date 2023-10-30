from pathlib import Path
import pandas as pd

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.propensities import simulate

data_dir = Path(__file__).parent.parent / 'data2' / 'fig2C'
data_dir.mkdir(exist_ok=True, parents=True)

channel_widths = (list(range(1,10,1)) + list(range(10,21,2)))[::-1]

simulate(
    n_sim=30000,
    channel_widths=channel_widths,
    results_file=data_dir / 'fig2C--propensities.csv',
    channel_length=300,
    n_workers=10,
    interval_after=2660,
    per_width_kwargs = {
        w: {
            'front_direction_minimal_distance': min(max(w - 1, 1), 5),
        } for w in channel_widths
    }
    )
