from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.binary import generate_dataset_batch
from scripts.analyze_binary import get_entropy


data_dir = Path(__file__).parent.parent / 'data'
data_dir.mkdir(parents=True, exist_ok=True)

channel_widths = []

for channel_length, intervals in {
    # 30: list(range(35, 110, 5)),
    # 52: list(range(70, 120, 5)),
    # 100: list(range(90, 135, 5)),
    # 170: list(range(100, 150, 5)),
    # 300: list(range(120, 165, 5)),
    # 300: list(range(145, 165, 5)),
    3: [1],
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
        processes=10,
    )
entropies = get_entropy(nearest_pulses.reset_index(), fields=['c'])
entropies.to_csv(data_dir / 'fig4B_entropies-c.csv')

