from pathlib import Path
import pandas as pd

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import generate_dataset_batch
from matplotlib import pyplot as plt

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1'  / 'approach4'
data_dir.mkdir(exist_ok=True, parents=True)

channel_lengths = [30]
# channel_widths = (list(range(1,10,1)) + list(range(10,21,2)))#[::-1]
channel_widths = [1]

dataset = generate_dataset_batch(
        channel_lengths,
        channel_widths,
        intervals=[300], 
        v=1/3.6,
        n_slots=5,
        n_simulations=1000,
        duration=1,
        p=1.,
        outdir=data_dir,
        n_margin=0,
        n_nearest=1,
        use_cached=False,
        processes=10,
    )

print(dataset[dataset['pulse_id'] == 2])
print((dataset[dataset['pulse_id'] == 2].groupby('channel_width')['c+0'].mean() + 30*3.6)/29)



