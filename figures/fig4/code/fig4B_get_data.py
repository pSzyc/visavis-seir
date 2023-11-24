from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.binary import generate_dataset_batch
from scripts.analyze_binary import get_entropy


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4'
data_dir.mkdir(parents=True, exist_ok=True)
    
channel_widths = []

for channel_length, intervals in {
    # 30: list(range(70, 131, 5)),
    # 52: list(range(75, 136, 5)),
    # 100: list(range(85, 146, 5)),
    # 170: list(range(100, 161, 5)),
    # 300: list(range(115, 176, 5)),
    # 520: list(range(135, 196, 5)),
    # 1000: list(range(170, 231, 5)),
    1000: list(range(195, 231, 5)),
    # 1700: list(range(205, 266, 5)),
    # 3000: list(range(255, 316, 5)),


    # 30: list(range(35, 110, 5)),
    # 52: list(range(70, 120, 5)),
    # 100: list(range(90, 135, 5)),
    # 170: list(range(100, 150, 5)),
    # 300: list(range(120, 165, 5)),
    # 300: list(range(145, 165, 5)),
    # 30: list(range(110,130,5)),
    # 52: list(range(120, 140, 5)),
    # 100: list(range(135, 155, 5)),
    # 1: [1],
}.items():

    nearest_pulses = generate_dataset_batch(
        channel_lengths=[channel_length],
        channel_widths=channel_widths,
        intervals=intervals,#[60,70,80,90,100],#list(range(110, 181, 5)), #[60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130, 150, 180, 220, 260, 300, 400], 
        outpath=data_dir / 'fig4B_nearest_pulses.csv',
        n_simulations=20,
        n_slots=250,
        n_margin=4,
        n_nearest=4,
        append=True,
        processes=1,
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
            entropies.to_csv(data_dir / f"fig4B_entropies-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}.csv")

