from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import generate_dataset_batch
from scripts.binary import get_entropy


data_dir = Path(__file__).parent.parent / 'data'/ 'fig4A' / 'approach3'
data_dir.mkdir(parents=True, exist_ok=True)

channel_widths = [6]
channel_lengths = [30, 100, 300, 1000]
# channel_lengths = [100, 300, 1000]
# channel_lengths = [1000]
intervals = list(range(20, 100, 10)) + list(range(100, 200, 5)) + list(range(200, 300, 20))
# intervals = [105, 185, 190, 195]
# intervals = list(range(35, 100, 10))
# intervals = []

nearest_pulses = generate_dataset_batch(
    channel_lengths=channel_lengths,
    channel_widths=channel_widths,
    intervals=intervals,#[60,70,80,90,100],#list(range(110, 181, 5)), #[60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130, 150, 180, 220, 260, 300, 400], 
    outpath=data_dir  / 'nearest_pulses.csv',
    # n_simulations=20,
    n_simulations=100,
    # n_slots=250,
    n_slots=500,
    duration=1,
    n_margin=4,
    n_nearest=4,
    append=False,
    processes=2,
)

fields_letter_to_fields = {
    'c': ['c+0'],
    'rl': ['l0', 'r0'],
    'cm': ['c+0', 'c-1'],
    'cp': ['c+0', 'c+1'],
    'cmp': ['c+0', 'c-1', 'c+1'],
}

for fields in 'c', 'rl', 'cm', 'cp', 'cmp':
    for k_neighbors in (15, 25):
        for reconstruction in (True, False):
            print(f"Estimating entropy ({fields}{k_neighbors}{'-reconstruction' if reconstruction else ''})")
            entropies = get_entropy(nearest_pulses.reset_index(), fields=fields_letter_to_fields[fields], reconstruction=reconstruction, k_neighbors=k_neighbors)
            entropies.to_csv(data_dir / f"fig4A_entropies-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}.csv")

