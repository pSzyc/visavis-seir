from pathlib import Path
import pandas as pd

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.two_fronts import simulate

data_dir = Path(__file__).parent.parent / 'data'
data_dir.mkdir(exist_ok=True, parents=True)

channel_widths = [6]#[::-1]
intervals = (list(range(100,300,100)))

simulate(
    n_sim=50,
    channel_widths=channel_widths,
    results_file=data_dir / 'fig3B--propensities.csv',
    channel_length=300,
    intervals=intervals,
    n_workers=1,
    interval_after=2660,
    per_width_kwargs = {
        w: {
            'front_direction_minimal_distance': min(max(w - 1, 1), 5),
            'min_peak_height': 0.03 / w,
        } for w in channel_widths
    }
    )