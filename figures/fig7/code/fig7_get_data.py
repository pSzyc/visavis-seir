# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

from pathlib import Path
import pandas as pd
from itertools import product

import sys

root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.two_fronts import get_pulse_fate_counts_batch
from scripts.defaults import PARAMETERS_DEFAULT

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig7' / 'fig7' / 'approach5'

channel_length = 300
channel_widths = [6, 15, 30, 60]#[::-1]


for altered_parameter, fold_changes in [
    ('r_forward_rate', [6/5, 6/6, 6/7, 6/8, 6/9]),
    ('e_forward_rate', [.7, .9, 1., 1.1, 1.3]),
    # ('e_forward_rate', [4/2, 4/3, 4/4, 4/5, 4/6]),
]:
    for fold_change in fold_changes:
        parameter_value = PARAMETERS_DEFAULT[altered_parameter] * fold_change
        tau_r = PARAMETERS_DEFAULT[f'r_subcompartments_count'] / (parameter_value if altered_parameter == 'r_forward_rate' else PARAMETERS_DEFAULT['r_forward_rate'])

        outdir = data_dir / f"{altered_parameter[0]}_rate" / f"{fold_change:.3f}"
        outdir.mkdir(exist_ok=True, parents=True)

        data = get_pulse_fate_counts_batch(
            n_simulations=3000,
            channel_widths=channel_widths,
            results_file=outdir / 'raw_probabilities.csv',
            channel_lengths=[channel_length],
            intervals=[4 * int(tau_r)],
            n_workers=20,
            interval_after=int(7.2*channel_length + 500),
            is_spawning_2='blocking',
            plot_results=False,
            save_iterations=False,
            use_cached=True,
            parameters={
                **PARAMETERS_DEFAULT,
                altered_parameter: parameter_value,
            }
            )


        possible_fates = ['anihilated', 'failure', 'lost_somewhere', 'transmitted']
        index_levels = ['channel_width', 'channel_length', 'interval']
        unique_index = [data.index.unique(level=level).tolist() for level in index_levels]

        grouped_data = (
            data.groupby(['fate', 'is_spawning'] + index_levels)
            [['count', 'first_event_position_sum', 'fronts_forward_sum', 'fronts_backward_sum']]
            .sum()
            .reindex(list(product(possible_fates, [0,1,2], *unique_index)))
            .fillna(0)
        )

        packed_data = pd.concat({
            'successful transmission': grouped_data.loc['transmitted', 0],
            'initiation failure': grouped_data.loc['lost_somewhere', 0] + grouped_data.loc['lost_somewhere', 1] + grouped_data.loc['lost_somewhere', 2],
            'propagation failure': grouped_data.loc['failure', 0],
            '<= 6 front spawning': grouped_data.loc['transmitted', 1] + grouped_data.loc['anihilated', 1] + grouped_data.loc['failure', 1],
            '> 6 front spawning': grouped_data.loc['transmitted', 2] + grouped_data.loc['anihilated', 2] + grouped_data.loc['failure', 2],
            'annihilation by backward front': grouped_data.loc['anihilated', 0],
        })
        packed_data = packed_data.rename_axis(['fate'] + packed_data.index.names[1:])

        all_started = packed_data.drop(index=['initiation failure'])
        total_started = all_started.groupby(index_levels).sum()
        events = all_started.drop(index=['successful transmission']).unstack('fate')
        events_avg_position = events['first_event_position_sum'].sum(axis=1) / events['count'].sum(axis=1)

        def compute_propensities(event_probabilities, average_distance_any_event, channel_length):
            total_event_probability = event_probabilities.sum(axis=1)
            return event_probabilities.div((total_event_probability * average_distance_any_event + (1 - total_event_probability) * channel_length), axis=0)


        probabilities = (packed_data['count'] / packed_data['count'].groupby(index_levels).sum()).unstack('fate')

        packed_data.to_csv(outdir / 'packed_data.csv')
        propensities.to_csv(outdir / 'propensities.csv')
        probabilities.to_csv(outdir / 'probabilities.csv')


