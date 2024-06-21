from pathlib import Path
import pandas as pd
import numpy as np

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.optimizer import find_maximum
from scripts.formula import get_expected_maximum_for_defaults
from scripts.analyze_binary import find_optimal_bitrate

# from matplotlib import pyplot as plt


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4CD' / 'approach6'
data_dir.mkdir(parents=True, exist_ok=True)

channel_widths = [6]
channel_lengths = [30,52,100,170,300,520,1000]


fields = 'c'
k_neighbors = 25
reconstruction = False
suffix = f"{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"

result_parts = []
entropies_parts = []

result, entropies = find_optimal_bitrate(
    get_expected_maximum_for_defaults(np.array(channel_lengths)), logstep=0.03, scan_points=5, 
    channel_lengths=channel_lengths,
    channel_widths=channel_widths,
    # outdir=data_dir / f'w-{channel_width}', 
    outdir=data_dir, 
    k_neighbors=k_neighbors, 
    reconstruction=reconstruction,
    fields=fields,
    n_slots=500, 
    n_simulations=100, #60 
    processes=10,
    use_cached=True,
    )
expected_maximums = result['optimal_interval']
result_parts.append(result)
entropies_parts.append(entropies)


result = pd.concat(result_parts).reset_index().drop_duplicates().set_index(['channel_width', 'channel_length']).sort_index()
result.to_csv(data_dir  / f'optimized_bitrate-{suffix}.csv')

entropies = pd.concat(entropies_parts)
entropies.to_csv(data_dir / f'entropies-{suffix}.csv')

            
