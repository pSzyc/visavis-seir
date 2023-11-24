from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.binary import generate_dataset_batch
from scripts.analyze_binary import get_entropy


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4'
data_dir.mkdir(parents=True, exist_ok=True)

channel_widths = [4,5,6,7,8,10,12]
channel_widths = [8,10,12]
channel_widths = [12]
channel_widths = []



for channel_length, (intervals, processes) in {
    # 30: (list(range(70, 181, 5)), 10),
    # 300: (list(range(90, 206, 5)), 4),
    300: (list(range(210, 220, 5)), 2),
    # 30: (list(range(60, 70, 5)), 20),

}.items():

    nearest_pulses = generate_dataset_batch(
        channel_lengths=[channel_length],
        channel_widths=channel_widths,
        intervals=intervals,
        outpath=data_dir / 'fig4C_nearest_pulses.csv',
        n_simulations=20,
        n_slots=250,
        n_margin=4,
        n_nearest=4,
        append=True,
        processes=processes,
    )   

fields_letter_to_fields = {
    'c': ['c'],
    'rl': ['l0', 'r0'],
    'cm': ['c', 'c-1'],
    'cp': ['c', 'c+1'],
    'cmp': ['c', 'c-1', 'c+1'],
}
for fields in 'c', 'rl', 'cm', 'cp', 'cmp':
    for k_neighbors in (15, 25):
        for reconstruction in (True, False):
            print(f"Estimating entropy ({fields}{k_neighbors}{'-reconstruction' if reconstruction else ''})")
            entropies = get_entropy(nearest_pulses.reset_index(), fields=fields_letter_to_fields[fields], reconstruction=reconstruction, k_neighbors=k_neighbors)
            entropies.to_csv(data_dir / f"fig4C_entropies-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}.csv")
