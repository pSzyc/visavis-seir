from pathlib import Path
import pandas as pd
from itertools import product

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.two_fronts import simulate

data_dir = Path(__file__).parent.parent / 'data' /'approach8'
data_dir.mkdir(exist_ok=True, parents=True)

channel_widths = [6]#[::-1]
intervals = [40,45,50,55,60,65,70,75,80,85,90,95,100,110,120,150,200,300]
channel_length = 1000

data = simulate(
    n_sim=3000,
    channel_widths=channel_widths,
    results_file=data_dir / 'raw_probabilities.csv',
    channel_length=channel_length,
    intervals=intervals,
    n_workers=20,
    interval_after=2660,
    plot_results=True,
    save_iterations=True,
    # use_cached=True,
    laptrack_parameters={
        "splitting_cost_cutoff": 14**2,
    },
    )



# data = pd.read_csv(data_dir / 'fig3B--probabilities.csv').set_index('interval')


# assert len(data['channel_length'].unique()) == 1, "More than one channel length found"
# channel_length = data['channel_length'].unique()[0]

possible_fates = ['transmitted', 'failure', 'anihilated', 'lost_somewhere']
intervals = data.index.get_level_values('interval').unique().tolist()

grouped_data = data.groupby(['interval', 'fate', 'is_spawning'])[['count', 'first_event_position_sum', 'fronts_forward_sum', 'fronts_backward_sum']].sum().reindex(list(product(intervals, possible_fates, [0,1,2]))).fillna(0)

packed_data = pd.concat({
    'successful transmission': grouped_data.loc[:, 'transmitted', 0],
    'fail to start': grouped_data.loc[:, 'lost_somewhere', 0] + grouped_data.loc[:, 'lost_somewhere', 1] + grouped_data.loc[:, 'lost_somewhere', 2],
    'propagation failure': grouped_data.loc[:, 'failure', 0],
    'one backward front spawning': grouped_data.loc[:, 'transmitted', 1] + grouped_data.loc[:, 'anihilated', 1] + grouped_data.loc[:, 'failure', 1],
    'other front spawning': grouped_data.loc[:, 'transmitted', 2] + grouped_data.loc[:, 'anihilated', 2] + grouped_data.loc[:, 'failure', 2],
    'anihilation by backward front': grouped_data.loc[:, 'anihilated', 0],
})
packed_data = packed_data.rename_axis(['fate'] + packed_data.index.names[1:])

all_started = packed_data.drop(index=['fail to start'])
total_started = all_started.groupby('interval').sum()
events = all_started.drop(index=['successful transmission']).unstack('fate')
events_avg_position = events['first_event_position_sum'].sum(axis=1) / events['count'].sum(axis=1)

def compute_propensities(event_probabilities, average_distance_any_event, channel_length):
    total_event_probability = event_probabilities.sum(axis=1)
    return event_probabilities.div((total_event_probability * average_distance_any_event + (1 - total_event_probability) * channel_length), axis=0)


propensities = compute_propensities(events['count'].div(total_started['count'], axis=0), events_avg_position, channel_length)
probabilities = (packed_data['count'] / packed_data['count'].groupby('interval').sum()).unstack('fate')

packed_data.to_csv(data_dir / 'packed_data.csv')
propensities.to_csv(data_dir / 'propensities.csv')
probabilities.to_csv(data_dir / 'probabilities.csv')


