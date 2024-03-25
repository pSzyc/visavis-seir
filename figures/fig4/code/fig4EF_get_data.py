from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import evaluation_fn
from scripts.optimizer import find_maximum

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig4' / 'fig4EF' / 'approach5'
data_dir.mkdir(parents=True, exist_ok=True)
    
channel_widths = [3,4,5,6,7,8,9,10,11,12]

channel_lengths = [100,300 1000] #,1000
# channel_lengths = [30] #,1000

channel_wls = list(product(channel_widths, channel_lengths))

def get_expected_maximum(channel_width, channel_length):
    return 3.13 * np.sqrt(channel_length) + 81.7  + 0.6*(np.log10(channel_length)-.3)*(channel_width-6)**2


fields_letter_to_fields = {
    'c': ['c+0'],
    'rl': ['l0', 'r0'],
    'cm': ['c+0', 'c-1'],
    'cp': ['c+0', 'c+1'],
    'cmp': ['c+0', 'c-1', 'c+1'],
}

fields = 'c'#, 'rl', 'cm', 'cp', 'cmp':
k_neighbors = 25
reconstruction = False
suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"
print(f"Estimating entropy {suffix}")


results_parts = {}
entropies = []
results = []
for channel_length in channel_lengths:
    x0 = np.log(get_expected_maximum(channel_widths[0], channel_length))
    for channel_width in channel_widths:
        if channel_length == 1000 and (channel_width < 4 or channel_width > 10): 
            continue
        # wrapped_eval_fn = store_results(evaluation_fn, results)
        log_opt_interval, max_bitrate = find_maximum(
            evaluation_fn, 
            x0=x0,
            xmin=np.log(30),
            xmax=np.log(400),
            x_tol=.4,
            prob_tol=1e-2,

            channel_width=channel_width,
            channel_length=channel_length,
            k_neighbors=k_neighbors,
            reconstruction=reconstruction,
            fields=fields_letter_to_fields[fields],
            suffix=suffix,
            outdir=data_dir,
            n_simulations=100,
            n_slots=500,
            n_margin=4,
            n_nearest=4,
            duration=1,
            min_distance_between_peaks=20,
            processes=5,
            use_cached=True,
            evaluation_logger=entropies,
        )
        print(np.exp(log_opt_interval), max_bitrate)
        # plt.plot(np.exp(log_opt_interval), max_bitrate, marker='x', color='red')

        optimal_interval = np.exp(log_opt_interval)


        results.append({
            'channel_width': channel_width,
            'channel_length': channel_length,
            'channel_length_sqrt': np.sqrt(channel_length),
            'optimal_interval': optimal_interval,
            'optimal_interval_sq': optimal_interval**2,
            'max_bitrate': max_bitrate,
            'max_bitrate_sq': max_bitrate**2,
            'max_bitrate_log': np.log(max_bitrate),
            'max_bitrate_inv': 1 / max_bitrate,
        })

        x0 = log_opt_interval

results = pd.DataFrame(results).set_index(['channel_width', 'channel_length'])
results.to_csv(data_dir / f'optimized_bitrate{suffix}.csv')
entropies = pd.concat(entropies).set_index(['channel_width', 'channel_length'])
entropies.to_csv(data_dir / f'entropies{suffix}.csv')

# result =  find_optimal_bitrate(expected_maximums, logstep=0.10, scan_points=11, outdir=data_dir / 'iteration1', k_neighbors=12, n_slots=200, n_simulations=20, processes=5) # was 20 for l=300
# expected_maximums = result['optimal_interval']
# result = find_optimal_bitrate(expected_maximums, logstep=0.06, scan_points=7, outdir=data_dir / 'iteration2', k_neighbors=18, n_slots=200, n_simulations=20, processes=5) # was 20 for l=300
# expected_maximums = result['optimal_interval']
# result = find_optimal_bitrate(expected_maximums, logstep=0.04, scan_points=5, outdir=data_dir / 'iteration3', k_neighbors=25, n_slots=500, n_simulations=60, processes=2) # was 10 for l=300


