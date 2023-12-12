from pathlib import Path
import pandas as pd
from itertools import product

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.two_fronts import simulate

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' /'approach5'
out_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3' /'approach1'
out_dir.mkdir(exist_ok=True, parents=True)

channel_widths = [6]#[::-1]
intervals = [40,45,50,55,60,65,70,75,80,85,90,95,100,110,120,150,200,300]
channel_lengths = [300]

n_simulations = 3000

pulse_fates = pd.concat(
    [
        pd.read_csv(data_dir / f'w-{channel_width}-l-{channel_length}' / f'interval-{interval}' / f'sim-{simulation_id}' / 'pulse_fates.csv')
        for channel_width, channel_length, interval, simulation_id in product(channel_widths, channel_lengths, intervals, range(n_simulations))
    ], 
    names=['channel_width', 'channel_length', 'interval', 'simulation_id'], 
    keys=product(channel_widths, channel_lengths, intervals, range(n_simulations)),
).rename(columns={0: 'pulse_id'}).reset_index().set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'pulse_id'])

print(pulse_fates)

selected_pulse_fates = pulse_fates[
        pulse_fates['fate'].eq('transmitted') 
        & pulse_fates['track_end_position'].gt(pulse_fates.index.get_level_values('channel_length') - 10)
    ]['track_end'].unstack('pulse_id').dropna()
print(selected_pulse_fates)

selected_pulse_fates.to_csv(out_dir / 'both_arrived_pulse_fates.csv')

mean_arrival_times = selected_pulse_fates.groupby(['channel_width', 'channel_length', 'interval']).mean()
mean_arrival_times.to_csv(out_dir / 'mean_arrival_times.csv')
median_arrival_times = selected_pulse_fates.groupby(['channel_width', 'channel_length', 'interval']).median()
median_arrival_times.to_csv(out_dir / 'median_arrival_times.csv')


print(mean_arrival_times)

