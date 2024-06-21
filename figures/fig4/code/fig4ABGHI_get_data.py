from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import generate_dataset_batch
from scripts.binary import get_entropy


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4AB' / 'approach7'# approach5
data_dir.mkdir(parents=True, exist_ok=True)
velocity_cache_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'velocity'



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
    outdir=data_dir,#  / 'nearest_pulses.csv',
    velocity_cache_dir=velocity_cache_dir,
    # n_simulations=20,
    n_simulations=100,
    # n_slots=250,
    n_slots=500,
    duration=1,
    n_margin=4,
    n_nearest=4,
    min_distance_between_peaks=30,
    use_cached=True,
    processes=10,
)

fields_letter_to_fields = {
    'c': ['c+0'],
    'rl': ['l0', 'r0'],
    'cm': ['c+0', 'c-1'],
    'cp': ['c+0', 'c+1'],
    'cmp': ['c+0', 'c-1', 'c+1'],
    'cmm': ['c+0', 'c-1', 'c-2'],
    'cmmp': ['c+0', 'c-1', 'c-2', 'c+1'],
}

for fields in 'cmmp',:#'c', 'rl', 'cm', 'cp', 'cmp', 'cmm', 'cmmp':
    for k_neighbors in (15, 25):
        for reconstruction in (True, False):
            suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"
            print(f"Estimating entropy {suffix}")

            entropies = get_entropy(nearest_pulses.reset_index(), fields=fields_letter_to_fields[fields], reconstruction=reconstruction, k_neighbors=k_neighbors)
            entropies.to_csv(data_dir / f"fig4AB_entropies{suffix}.csv")

