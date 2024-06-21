from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.formula import get_expected_maximum_for_defaults
from scripts.analyze_binary import find_optimal_bitrate

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4EF' / 'approach7'
data_dir.mkdir(parents=True, exist_ok=True)
    
channel_widths_s = (
    [6,5,4,3],
    [6,7,8,9,10,11,12],
    )

channel_lengths = [100,300,1000] #,1000


fields = 'c'
k_neighbors = 25
reconstruction = False
suffix = f"{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"

result_parts = []
entropies_parts = []

for channel_widths in channel_widths_s:
    expected_maximums = get_expected_maximum_for_defaults(np.array(channel_lengths))
    for channel_width in channel_widths:
        result, entropies = find_optimal_bitrate(
            expected_maximums, logstep=0.03, scan_points=5, 
            channel_lengths=channel_lengths,
            channel_widths=[channel_width],
            outdir=data_dir / f'w-{channel_width}', 
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
result.to_csv((data_dir / suffix) / f'optimized_bitrate-{suffix}.csv')

entropies = pd.concat(entropies_parts)
entropies.to_csv(data_dir / f'entropies-{suffix}.csv')

