from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.binary import generate_dataset_batch
from scripts.analyze_binary import get_entropy


data_dir = Path(__file__).parent.parent / 'data'
data_dir.mkdir(parents=True, exist_ok=True)

channel_widths = [6]
channel_lengths = [30, 100, 300, 1000]
# channel_lengths = [1000]
# intervals = list(range(20, 110, 10)) + list(range(110, 180, 5)) + list(range(180, 300, 20))
intervals = []

nearest_pulses = generate_dataset_batch(
    channel_lengths=channel_lengths,
    channel_widths=channel_widths,
    intervals=intervals,#[60,70,80,90,100],#list(range(110, 181, 5)), #[60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130, 150, 180, 220, 260, 300, 400], 
    outpath=data_dir / 'fig4A_nearest_pulses1.csv',
    n_simulations=20,
    n_slots=250,
    n_margin=4,
    n_nearest=4,
    append=True,
    processes=20,
)
entropies = get_entropy(nearest_pulses.reset_index(), fields=['c'], reconstruction=False, k_neighbors=25)
entropies.to_csv(data_dir / 'fig4A_entropies1-c25.csv')

