from pathlib import Path
import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3'/ 'figS3-3' / 'approach1'

from scripts.propensities import simulate

for channel_length in [30, 100, 300, 1000]:
    data = simulate(
        n_sim=5,
        channel_widths=[6],
        channel_length=channel_length,
        results_file= data_dir / 'propensities.csv',
        n_workers=20,
        interval_after=channel_length * 4,
        per_width_kwargs = {
            w: {
                'front_direction_minimal_distance': min(max(w - 1, 1), 5),
                'min_peak_height': 0.03 / w,
            } for w in [6]
        },
    )