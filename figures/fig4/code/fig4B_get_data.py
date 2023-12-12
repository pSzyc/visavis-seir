from pathlib import Path
import pandas as pd
import numpy as np

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import generate_dataset_batch
from scripts.binary import get_entropy, get_optimal_bitrate


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4B' / 'approach1'
data_dir.mkdir(parents=True, exist_ok=True)
    
channel_widths = [6]

channel_lengths = [30,52,100,170,300,520,1000]
# channel_lengths = [1700]



def get_expected_maximum(channel_length):
    return 3.13 * np.sqrt(channel_length) + 81.7

expected_maximums = get_expected_maximum(np.array(channel_lengths))# * 1.1
interval_scan_steps = expec`ted_maximums // 30
interval_scan_centers = expected_maximums // 1
scan_points = 3
scan_ranges = np.linspace(
        interval_scan_centers - scan_points * interval_scan_steps, 
        interval_scan_centers + scan_points * interval_scan_steps,
        2 * scan_points + 1, dtype='int').T

nearest_pulses = pd.concat([
    generate_dataset_batch(
        channel_lengths=[channel_length],
        channel_widths=channel_widths,
        intervals=intervals,
        outdir=data_dir,
        n_simulations=100,
        n_slots=500,
        n_margin=4,
        n_nearest=4,
        duration=1,
        append=True,
        processes=2,
        use_cached=True,
    ) for channel_length, intervals in zip(channel_lengths, scan_ranges)
], ignore_index=True)

fields_letter_to_fields = {
    'c': ['c+0'],
    'rl': ['l0', 'r0'],
    'cm': ['c+0', 'c-1'],
    'cp': ['c+0', 'c+1'],
    'cmp': ['c+0', 'c-1', 'c+1'],
}
for fields in 'c',:#, 'rl', 'cm', 'cp', 'cmp':
    for k_neighbors in (25,):
        for reconstruction in (False,):
            suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"
            print(f"Estimating entropy {suffix}")

            entropies = get_entropy(nearest_pulses.reset_index(), fields=fields_letter_to_fields[fields], reconstruction=reconstruction, k_neighbors=k_neighbors)
            entropies.to_csv(data_dir / f"entropies{suffix}.csv")

            result = get_optimal_bitrate(entropies, outdir=data_dir)
