from pathlib import Path
import pandas as pd

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.two_fronts import simulate

data_dir = Path(__file__).parent.parent / 'data'
data_dir.mkdir(exist_ok=True, parents=True)

channel_widths = [6]#[::-1]
intervals = (list(range(60,300,100)))

simulate(
    n_sim=5,
    channel_widths=channel_widths,
    results_file=data_dir / 'fig3B--probabilites.csv',
    channel_length=300,
    intervals=intervals,
    n_workers=1,
    interval_after=2660,
    )
