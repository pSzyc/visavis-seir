# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.formula import get_expected_maximum_for_defaults
from scripts.analyze_binary import find_optimal_bitrate

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4EF' / 'approach10'#'approach9--500-rep--100-pulses'
data_dir.mkdir(parents=True, exist_ok=True)
    
channel_widths_s = (
    [6,5,4,3,2,1],
    # [6,7,8,9,10,11,12] + list(range(15,31,3)),
    list(range(6,31,1)),
    # list(range(6,13,1)),
    # list(range(6,21,1)),
    )

channel_lengths = [30,100,300,1000] #,1000


fields = 'c'
k_neighbors = 25
reconstruction = False
suffix = f"{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"

result_parts = []
entropies_parts = []

for channel_widths in channel_widths_s:
    expected_maximums = get_expected_maximum_for_defaults(np.array(channel_lengths))
    channel_lengths_worth_progressing = channel_lengths
    result_prev = None
    for channel_width in channel_widths:
        result, entropies = find_optimal_bitrate(
            expected_maximums, logstep=0.03, scan_points=5, 
            channel_lengths=channel_lengths_worth_progressing,
            channel_widths=[channel_width],
            outdir=data_dir / f'w-{channel_width}', 
            k_neighbors=k_neighbors, 
            reconstruction=reconstruction,
            fields=fields,
            n_slots=500, #500
            n_simulations=100, #100 
            processes=20,
            use_cached=True,
            )
        # expected_maximums = result['optimal_interval']
        result_parts.append(result)
        entropies_parts.append(entropies)

        print(result_prev is None)
        if result_prev is None:
            worth_progressing_condition = result['max_bitrate'] > 0.02 / 60
        else:
            worth_progressing_condition = (
                (result['max_bitrate'] > 0.02 / 60)
              & (result_prev[result_prev.index.get_level_values('channel_length').isin(channel_lengths_worth_progressing)].reset_index('channel_width')['max_bitrate'] < 2 * result.reset_index('channel_width')['max_bitrate']).to_numpy()
            )
            print((result['max_bitrate'] > 0.02 / 60))
            print((result_prev[result_prev.index.get_level_values('channel_length').isin(channel_lengths_worth_progressing)].reset_index('channel_width')['max_bitrate'] < 2 * result.reset_index('channel_width')['max_bitrate']).to_numpy())
            print(worth_progressing_condition)

        worth_progressing = result[worth_progressing_condition]
        channel_lengths_worth_progressing = worth_progressing.index.get_level_values('channel_length').tolist()
        expected_maximums = worth_progressing['optimal_interval']

        result_prev = result

        if not len(channel_lengths_worth_progressing):
            break


result = pd.concat(result_parts).reset_index().drop_duplicates().set_index(['channel_width', 'channel_length']).sort_index()
result.to_csv(data_dir / f'optimized_bitrate-{suffix}.csv')

entropies = pd.concat(entropies_parts)
entropies.to_csv(data_dir / f'entropies-{suffix}.csv')
