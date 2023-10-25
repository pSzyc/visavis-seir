from pathlib import Path
import pandas as pd

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_tracking import generate_dataset
from scripts.utils import starmap


data_dir = Path(__file__).parent.parent / 'data'
data_dir.mkdir(parents=True, exist_ok=True)
 

outdir = data_dir / 'fig2C'
outdir.mkdir(exist_ok=True, parents=True)

input_protocol = []

channel_lengths = [300]
# channel_widths = list(range(1,10)) + list(range(10,21,2))
channel_widths = [1,2,3,4,5,6,7,8,9] + list(range(10,21,2))

data_parts = starmap(generate_dataset, [
        dict(
            input_protocol=input_protocol,
            n_simulations=10,
            channel_length=channel_length,
            channel_width=channel_width,
            outdir=outdir / f"w-{channel_width}-l-{channel_length}",
            n_margin=0,
            interval_after=int(2.2 * channel_length * 3.6),
            plot_results=True,
            save_states=False,
            save_iterations=True,
            front_direction_minimal_distance=min(channel_width, 5),
        )
    for channel_length in channel_lengths for channel_width in channel_widths
    ], processes=20)

pulse_fates = pd.concat([data_part.set_index(['channel_length', 'channel_width', 'simulation_id']) for data_part in data_parts])
pulse_fates.to_csv(outdir / 'pulse_fates.csv')

