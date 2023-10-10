import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from subplots_from_axsize import subplots_from_axsize
from scipy.signal import find_peaks
from typing import Iterable
from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors


from laptrack import LapTrack


import sys
sys.path.insert(0, '..') # in order to be able to import from scripts.py

from scripts.plot_result import plot_result_from_states


def get_pulse_positions(data, min_distance=6):

    data['act'] = 1.0 * ((data['E'] > 0) | (data['I'] > 0))

    data_grouped = data.groupby(['seconds', 'h'], sort=True).mean()

    raw_data_df = data_grouped['act']
    raw_data = raw_data_df.unstack('h').to_numpy()
    # plt.figure(figsize=(100,10))
    # plt.imshow(raw_data, cmap='binary')
    # plt.show()

    smoothed_data = gaussian_filter(raw_data, sigma=[2., 2.5], mode='nearest', truncate=2.01)
    # plt.figure(figsize=(100,10))
    # plt.imshow(smoothed_data, cmap='binary')
    data_grouped['act'] = pd.DataFrame(smoothed_data).stack().to_numpy()

    # plt.figure(figsize=(100,10))
    # plt.imshow(data_grouped['act'].unstack('h').to_numpy(), cmap='binary')
    # plt.show()
    # return


    pulse_positions = []
    for seconds, data_slice in data_grouped.groupby('seconds'):
        # data_slice = data_time.groupby('h').mean()
        pulse_positions_time, _ = find_peaks(data_slice['act'], distance=min_distance)
        pulse_positions.extend(
            {   
                'seconds': seconds,
                'h': h,
            } for h in pulse_positions_time
        )
        
    pulse_positions = pd.DataFrame(pulse_positions)
    return pulse_positions


def get_tracks(pulse_positions, duration):

    lt = LapTrack(
        track_dist_metric="sqeuclidean",  
        splitting_dist_metric="sqeuclidean",
        merging_dist_metric="sqeuclidean",
        
        track_cost_cutoff=3**2,
        splitting_cost_cutoff=14**2,
        merging_cost_cutoff=False,  # no merging
        gap_closing_max_frame_count=4,
        gap_closing_cost_cutoff=5**2,
        
        segment_start_cost=3000,#10**2,
        segment_end_cost=3000,#10**2,
        track_start_cost=300,#20**2,
        track_end_cost=300,#20**2,
        
        no_splitting_cost=0,
        no_merging_cost=0,
    )
    pulse_positions['frame'] = pulse_positions['seconds'] // duration

    track_df, split_df, merge_df = lt.predict_dataframe(
        pulse_positions,
        coordinate_cols=['h'],
        frame_col='frame',
        only_coordinate_cols=False,
        validate_frame=False,
    )
    tracks = track_df.reset_index(drop=True).rename(columns={'frame_y': 'frame'})
   
    return tracks, split_df, merge_df


def get_track_info(tracks, split_df, front_direction_minimal_distance=5, min_track_length=5):

    enhanced_split_df = split_df.copy()
    enhanced_split_df['child_no'] = np.arange(len(split_df)) % 2
    enhanced_split_df = enhanced_split_df.set_index(['parent_track_id', 'child_no'])['child_track_id'].unstack('child_no').rename(columns={0: 'first_child_track_id', 1: 'second_child_track_id'})
    
    track_info = (tracks.groupby('track_id')
                ['seconds'].agg(['size', 'min', 'max']).rename(columns={'size': 'track_length', 'min': 'track_start', 'max': 'track_end'})
                .join(tracks[['track_id', 'tree_id']].drop_duplicates().set_index('track_id'), how='left')
                .join(enhanced_split_df)
    )
    track_info.index.name = 'track_id'

    track_info['track_start_position'] = tracks.set_index(['track_id', 'seconds']).reindex(
        list(zip(track_info.index, track_info['track_start']))
        ).reset_index('seconds')['h']
    track_info['track_end_position'] = tracks.set_index(['track_id', 'seconds']).reindex(
        list(zip(track_info.index, track_info['track_end']))
        ).reset_index('seconds')['h']
    
    track_info['front_direction'] = pd.Series({
        track_id:
            1 * (track['h'].iloc[-1] - track['h'].iloc[0] >= front_direction_minimal_distance)
            - 1 * (track['h'].iloc[-1] - track['h'].iloc[0] <= -front_direction_minimal_distance)
            for track_id, track in tracks.groupby('track_id')})

    good_tracks = track_info.index.unique()
    previous_good_tracks = []
    while len(previous_good_tracks) != len(good_tracks):
        track_info['is_first_child_good'] = track_info['first_child_track_id'].isin(good_tracks)
        track_info['is_second_child_good'] = track_info['second_child_track_id'].isin(good_tracks)
        previous_good_tracks = good_tracks
        good_tracks = track_info[
            (track_info['track_length'] >= min_track_length)
            | track_info['is_first_child_good']
            | track_info['is_second_child_good']
            # | track_info['track_end_position'].gt(channel_length - CHANNEL_END_TOLERANCE)
            ].index.unique()

    track_info['is_good_track'] = track_info.index.isin(good_tracks)

    return track_info


def get_front_fates(tracks, track_info, channel_length, v, channel_end_tolerance=8, te_back=20, te_forward=10, te_space=10):
    fates = track_info[
        track_info['is_good_track']
        # & track_info['front_direction'].eq(1)
        & ~track_info['is_first_child_good']
        & ~track_info['is_second_child_good']
        ].copy()

    has_near_neighbor = pd.Series(
    [(
        tracks['track_id'].ne(track_id) 
        & tracks['track_id'].ne(track_info.loc[track_id]['first_child_track_id'])
        & tracks['track_id'].ne(track_info.loc[track_id]['second_child_track_id'])
        & tracks['seconds'].between(row['track_end'] - te_back, row['track_end'] + te_forward)
        & tracks['h'].between(row['track_end_position'] - te_space, row['track_end_position'] + te_space)
        ).any()
        for track_id, row in fates.iterrows()
    ], index=fates.index)

    nbrs = NearestNeighbors(n_neighbors=3)
    nbrs.fit(list(zip(fates['track_end'] * v, fates['track_end_position'])))
    nearest_endings = nbrs.radius_neighbors(radius=25, return_distance=False)
    front_directions = fates['front_direction'].to_numpy()
    has_near_ending = pd.Series([
        any(direction * front_directions[other_ending] == -1 for other_ending in nes) 
        for direction, nes in zip(front_directions, nearest_endings)
        ], index=fates.index) 

    fates['fate'] = 'failure' # note that this can be overriden
    fates['fate'] = fates['fate'].mask(has_near_neighbor | has_near_ending, 'anihilated') # note that this can be overriden
    fates['fate'] = fates['fate'].mask(~has_near_neighbor & fates['track_end_position'].ge(channel_length - channel_end_tolerance) & fates['front_direction'].eq(1), 'transmitted')
    fates['fate'] = fates['fate'].mask(~has_near_neighbor & fates['track_end_position'].lt(channel_end_tolerance) & fates['front_direction'].eq(-1), 'transmitted')

    return fates.sort_values(['tree_id', 'track_end'])



def plot_tracks(tracks, track_info, fates, outpath=None, show=True, panel_size=(8,4)):
    fig, axs = subplots_from_axsize(2, 1, panel_size)

    good_tracks = track_info[track_info['is_good_track']].index

    tracks_selected = tracks[tracks['track_id'].isin(good_tracks)]#[(tracks['track_id'] >= 0) & (tracks['track_id'] <= 31)]
    tracks_not_selected = tracks[~tracks['track_id'].isin(good_tracks)]#[(tracks['track_id'] >= 0) & (tracks['track_id'] <= 31)]


    axs[0].scatter(
        tracks_selected['seconds'],
        tracks_selected['h'],
        c=tracks_selected['track_id'],
        s=5,
        cmap='prism'
    )

    axs[0].scatter(
        tracks_not_selected['seconds'],
        tracks_not_selected['h'],
        c='k',
        s=5,
    )

    for track_id, track in tracks_selected.groupby('track_id'):
        axs[0].plot(track['seconds'], track['h'], color='k', alpha=0.4, lw=1)

    for fate, color in (
        ('transmitted', 'blue'),
        ('failure', 'green'),
        ('anihilated', 'red'),
    ):
        events = fates[fates['fate'] == fate]
        axs[0].scatter(
            events['track_end'],
            events['track_end_position'],
            marker='x',
            color=color,
        )


    axs[1].scatter(
        tracks_selected['seconds'],
        tracks_selected['h'],
        c=tracks_selected['tree_id'],
        s=5,
        cmap='prism',
    )
    if outpath is not None:
        plt.savefig(outpath)
    if show:
        plt.show()



def plot_kymograph_with_endings(data, fates, duration,outpath=None, show=True, panel_size=(10, 4)):

    fig, ax = plot_result_from_states(data, show=False, panel_size=panel_size)

    for fate, color in (
        ('transmitted', 'blue'),
        ('failure', 'green'),
        ('anihilated', 'red'),
    ):
        events = fates[fates['fate'] == fate]
        ax.scatter(
            events['track_end'] / duration,
            events['track_end_position'],
            marker='x',
            color=color,
        )

    if outpath is not None:
        plt.savefig(outpath)

    if show:
        plt.show()



def get_input_pulse_to_tree_id(tracks, pulse_times):
    input_pulse_to_tree_id = [
        
        (lambda x: int(x.iloc[0]['tree_id']) if len(x) == 1 else 
            print(f"WARNING: {len(x)} pulse starts found near channel start at time {input_pulse}.") or np.nan)(
            tracks[tracks['seconds'].eq(input_pulse) & tracks['h'].le(3)]
        )
            
            for input_pulse in pulse_times
    ]

    pulse_id = -1
    for i in range(len(input_pulse_to_tree_id)):
        if input_pulse_to_tree_id[i] == pulse_id:
            input_pulse_to_tree_id[i] = np.nan
        else:
            pulse_id = input_pulse_to_tree_id[i]
    return input_pulse_to_tree_id


def get_pulse_fates(front_fates: pd.DataFrame, input_pulse_to_tree_id, v):
    front_fates = front_fates.assign(timespace= lambda x: x['track_end'] - x['track_end_position'] / v).sort_values(['tree_id', 'timespace'])

    pulse_fates = pd.Series([
        ('lost_somewhere' if len(arrivals[arrivals['front_direction'] == 1]) == 0
        else arrivals[arrivals['front_direction'] == 1]['fate'].iloc[0]
        )
        
        for tree_id in input_pulse_to_tree_id
        for arrivals in [
            front_fates[front_fates['tree_id'] == tree_id]
            ]
    ])
    pulse_fates.name = 'fate'

    spawning_counts = pd.DataFrame([{
        'forward': (arrivals['front_direction'] == 1).sum() - 1,
        'short': (arrivals['front_direction'] == 0).sum(),
        'backward': (arrivals['front_direction'] == -1).sum(),
    } 
        for tree_id in input_pulse_to_tree_id
        for arrivals in [
            front_fates[front_fates['tree_id'] == tree_id]
            ]
    ])


    pulse_fates_and_spawned = pd.concat([pulse_fates, spawning_counts], axis='columns')

    return pulse_fates_and_spawned

## ---------

def determine_fates(data: pd.DataFrame, input_protocol: Iterable[float], v=1/3.6, outpath=None, plot_results=False):

    if outpath is not None:
        outpath = Path(outpath)

    duration = data['seconds'].drop_duplicates().sort_values().diff()[0]
    channel_length = data['h'].max() + 1

    print('Determining pulse positions...')
    pulse_positions = get_pulse_positions(data)
    if outpath is not None:
        pulse_positions.to_csv(outpath / 'pulse_positions.csv')

    print('Tracking...')
    tracks, split_df, merge_df = get_tracks(pulse_positions, duration)
    if outpath is not None:
        tracks.to_csv(outpath / 'tracks.csv')

    print('Extracting track information...')
    track_info = get_track_info(tracks, split_df)
    if outpath is not None:
        track_info.to_csv(outpath / 'track_info.csv')

    print('Determining front fates...')
    front_fates = get_front_fates(tracks, track_info, channel_length, v)
    if outpath is not None:
        front_fates.to_csv(outpath / 'fates.csv')

    if plot_results:
        print("Plotting tracks...")
        panel_size = (tracks['frame'].max() / 100, (tracks['h'].max() + 1) / 100)
        plot_tracks(tracks, track_info, front_fates, show=False, outpath=(outpath / 'out.svg') if outpath else None, panel_size=panel_size)
        plot_kymograph_with_endings(data, front_fates, duration, show=False, outpath=(outpath / 'out-kymo.png') if outpath else None, panel_size=panel_size)


    print('Matching input pulses with trees...')
    pulse_times = [0] + list(np.cumsum(input_protocol))[:-1]
    input_pulse_to_tree_id = get_input_pulse_to_tree_id(tracks, pulse_times)
    if outpath is not None:
        pd.Series(input_pulse_to_tree_id).to_csv(outpath / 'input_pulse_to_tree_id.csv')

    print('Determining pulse fates...')
    pulse_fates = get_pulse_fates(front_fates, input_pulse_to_tree_id, v)
    if outpath is not None:
        pulse_fates.to_csv(outpath / 'pulse_fates.csv')

    print('Done.')
    return pulse_fates





