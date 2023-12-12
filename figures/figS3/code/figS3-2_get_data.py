from pathlib import Path
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' /'approach8'
out_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3-2' /'approach2'
out_dir.mkdir(exist_ok=True, parents=True)

channel_widths = [6]#[::-1]
intervals = [40,45,50,55,60,65,70,75,80,85,90,95,100,110,120,150,200,300]
channel_lengths = [1000]

n_simulations = 300



front_fates = pd.concat(
    [
        pd.read_csv(data_dir / f'w-{channel_width}-l-{channel_length}' / f'interval-{interval}' / f'sim-{simulation_id}' / 'front_fates.csv').pipe(
            lambda df: df[df['fate'].eq('transmitted') & df['front_direction'].eq(1) & df['tree_id'].eq(2)]
        ).reset_index(drop=True).reset_index()
        for channel_width, channel_length, interval, simulation_id in product(channel_widths, channel_lengths, intervals, range(n_simulations))
    ], 
    names=['channel_width', 'channel_length', 'interval', 'simulation_id'], 
    keys=product(channel_widths, channel_lengths, intervals, range(n_simulations)),
).rename(columns={'index': 'arrival_id'}).reset_index().set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'arrival_id'])

print(front_fates)


unstacked_front_fates = front_fates['track_end'].unstack('arrival_id')
spawned_front_fates = unstacked_front_fates[unstacked_front_fates.notna().sum(axis=1) >= 2]

spawned_front_fates.to_csv(out_dir / 'spawned_front_fates.csv')

# selected_front_fates.plot.hist(bins=range(1100,1650,10), alpha=0.3)


# counts = selected_front_fates.groupby(['channel_width', 'channel_length', 'interval', 'arrival_id', 'track_end']).size()
# counts.name = 'count'
# counts = counts.reset_index()
# plt.scatter(counts['track_end'], counts['interval'], c=counts['arrival_id'], s=counts['count'])


# selected_front_fates = front_fates[
#         front_fates['fate'].eq('transmitted') 
#         & front_fates['track_end_position'].gt(front_fates.index.get_level_values('channel_length') - 10)
#     ]
# print(selected_pulse_fates)

# mean_arrival_times = selected_pulse_fates.groupby(['channel_width', 'channel_length', 'interval']).mean()
# print(mean_arrival_times)

# mean_arrival_times.to_csv(out_dir / 'mean_arrival_times.csv')

# plt.savefig(out_dir / 'trial.svg')
# plt.savefig(out_dir / 'trial.png')


