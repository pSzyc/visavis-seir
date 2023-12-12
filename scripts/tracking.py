import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from subplots_from_axsize import subplots_from_axsize
from scipy.signal import find_peaks
from typing import Iterable
from pathlib import Path
from scipy.ndimage import gaussian_filter,gaussian_filter1d, convolve1d
from sklearn.neighbors import NearestNeighbors
import json

from laptrack import LapTrack

import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) # in order to be able to import from scripts.py

from scripts.plot_result import plot_result_from_activity

fate_to_color = {
    'anihilated': 'red',
    'failure': 'green',
    'transmitted': 'blue',
    'lost_somewhere': 'purple',
}



def get_pulse_positions(activity, min_distance_between_peaks=6, smoothing_sigma_h=1.5, smoothing_sigma_frames=1.2, min_peak_height=0.002, exp_mean=.8): #2., 1.8

    exp_kernel_size = int(3 * exp_mean) + 1
    exp_kernel = np.exp(-np.arange(0, exp_kernel_size) / exp_mean)
    exp_kernel = exp_kernel / exp_kernel.sum()

    # smoothed_data = gaussian_filter(1.*activity.to_numpy(), sigma=[smoothing_sigma_frames, smoothing_sigma_h], mode='nearest', truncate=2.01)
    smoothed_data = convolve1d(
        gaussian_filter1d(
            1.*activity.to_numpy(), sigma=smoothing_sigma_h, axis=1, mode='constant', truncate=2.01
        ),
        exp_kernel,
        origin=-(exp_kernel_size // 2),
        mode='nearest',
        axis=0,
    )

    pulse_positions = []
    for seconds, activity_slice in zip(activity.index.get_level_values('seconds'), smoothed_data):
        # data_slice = data_time.groupby('h').mean()
        pulse_positions_part, _ = find_peaks(activity_slice, distance=min_distance_between_peaks, height=min_peak_height)
        pulse_positions.extend(
            {
                'seconds': seconds,
                'h': h,
            } for h in [-200] + list(pulse_positions_part) # -200 is an artificial maximum added so that LapTrack does not omit emtpy frames
        )
    
    pulse_positions = pd.DataFrame(pulse_positions)
    return pulse_positions


def get_tracks(pulse_positions, duration, laptrack_parameters={}):

    lt = LapTrack(
        **{**dict(
            track_dist_metric="sqeuclidean",
            splitting_dist_metric="sqeuclidean",
            merging_dist_metric="sqeuclidean",
            
            track_cost_cutoff=5**2,
            splitting_cost_cutoff=14**2,
            merging_cost_cutoff=False,  # no merging
            gap_closing_max_frame_count=0,
            gap_closing_cost_cutoff=0,
            
            segment_start_cost=3000,#10**2,
            segment_end_cost=3000,#10**2,
            track_start_cost=300,#20**2,
            track_end_cost=300,#20**2,
            
            no_splitting_cost=0,
            no_merging_cost=0,
        ),
        **laptrack_parameters
        }
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
    tracks = tracks[tracks['h'] != -200] # remove the artificial track
   
    return tracks, split_df, merge_df


def get_track_info(tracks, split_df, pulse_times, channel_length, v, front_direction_minimal_distance=5, min_track_length=5, min_significant_track_length=10,  channel_end_tolerance=8):


    track_info = (tracks.groupby('track_id')
                ['seconds'].agg(['size', 'min', 'max']).rename(columns={'size': 'track_length', 'min': 'track_start', 'max': 'track_end'})
                .join(tracks[['track_id', 'tree_id']].drop_duplicates().set_index('track_id'), how='left')
    )
    track_info.index.name = 'track_id'

    track_info['track_start_position'] = tracks.set_index(['track_id', 'seconds']).reindex(
        list(zip(track_info.index, track_info['track_start']))
        ).reset_index('seconds')['h']
    track_info['track_end_position'] = tracks.set_index(['track_id', 'seconds']).reindex(
        list(zip(track_info.index, track_info['track_end']))
        ).reset_index('seconds')['h']

    def assign_tree_id_recursively(track_id, tree_id):
        track_info['tree_id'][track_id] = tree_id
        if track_id in enhanced_split_df.index:
            first_child, second_child = enhanced_split_df.loc[track_id]
            if not np.isnan(first_child):
                assign_tree_id_recursively(int(first_child), tree_id)
            if not np.isnan(second_child):
                assign_tree_id_recursively(int(second_child), tree_id)

    def merge_tracks(parent_track_id, child_track_id):
        tracks['track_id'].replace(child_track_id, parent_track_id, inplace=True)
        track_info.loc[parent_track_id, ['track_end', 'track_end_position']] = track_info.loc[child_track_id, ['track_end', 'track_end_position']]
        track_info.loc[parent_track_id, 'track_length'] = track_info.loc[parent_track_id, 'track_length'] + track_info.loc[child_track_id]['track_length']
        track_info.drop(index=child_track_id, inplace=True)
        if child_track_id in enhanced_split_df.index:
            enhanced_split_df.loc[parent_track_id] = enhanced_split_df.loc[child_track_id]
            enhanced_split_df.drop(index=child_track_id, inplace=True)
        else:
            enhanced_split_df.drop(index=parent_track_id, inplace=True)

    n_trees = track_info['tree_id'].max() + 1
    
    enhanced_split_df = split_df.copy()
    enhanced_split_df['child_no'] = np.arange(len(split_df)) % 2
    enhanced_split_df = enhanced_split_df.set_index(['parent_track_id', 'child_no'])['child_track_id'].unstack('child_no').rename(columns={0: 'first_child_track_id', 1: 'second_child_track_id'}).reindex(columns=['first_child_track_id', 'second_child_track_id'])
    

    track_info_with_children = (
        track_info
            .join(enhanced_split_df)
            .join(track_info, on='first_child_track_id', rsuffix='_first_child')
            .join(track_info, on='second_child_track_id', rsuffix='_second_child')
    )

    tracks_with_first_child_at_pulse = track_info_with_children[
        track_info_with_children['track_start_position_first_child'].le(3)
        & track_info_with_children['track_start_first_child'].isin(pulse_times)
    ]

    for parent_track_id, track in tracks_with_first_child_at_pulse.iterrows():
        track_id = track['first_child_track_id']
        sibling_track_id = track['second_child_track_id']
        merge_tracks(parent_track_id, sibling_track_id)
        assign_tree_id_recursively(track_id, n_trees)
        n_trees += 1

    tracks_with_second_child_at_pulse = track_info_with_children[
        track_info_with_children['track_start_position_second_child'].le(3)
        & track_info_with_children['track_start_second_child'].isin(pulse_times)
    ]

    for parent_track_id, track in tracks_with_second_child_at_pulse.iterrows():
        track_id = track['second_child_track_id']
        sibling_track_id = track['first_child_track_id']
        merge_tracks(parent_track_id, sibling_track_id)
        assign_tree_id_recursively(track_id, n_trees)
        n_trees += 1

    tracks['tree_id'] = tracks.join(track_info, on='track_id', lsuffix='_old')['tree_id']
    track_info = track_info.join(enhanced_split_df)
    first_children = track_info['first_child_track_id'].dropna()
    second_children = track_info['second_child_track_id'].dropna()
    track_info.loc[first_children, 'parent_track_id'] = first_children.index
    track_info.loc[first_children, 'sibling_track_id'] = second_children.to_numpy()
    track_info.loc[second_children, 'parent_track_id'] = second_children.index
    track_info.loc[second_children, 'sibling_track_id'] = first_children.to_numpy()


    track_info['front_direction'] = (
        1 *   (  track_info['track_end_position'] - track_info['track_start_position']).ge(
            np.minimum(front_direction_minimal_distance, 0.5 * v * (track_info['track_end'] - track_info['track_start']))) 
        - 1 * (-(track_info['track_end_position'] - track_info['track_start_position'])).ge(
            np.minimum(front_direction_minimal_distance, 0.5 * v * (track_info['track_end'] - track_info['track_start']))) 
    )
            # 1 * ((track['h'].iloc[-1] - track['h'].iloc[0] >= front_direction_minimal_distance) | (track['h'].iloc[-1] - track['h'].iloc[0] >= 0.8 * v * (track['seconds'].iloc[-1] - track['seconds'].iloc[0] + 1)))
            # - 1 * ((track['h'].iloc[-1] - track['h'].iloc[0] <= -front_direction_minimal_distance) | (track['h'].iloc[-1] - track['h'].iloc[0] <= -0.8 * v * (track['seconds'].iloc[-1] - track['seconds'].iloc[0] + 1)))
            # for track_id, track in tracks.groupby('track_id')})


    good_tracks = track_info.index.unique()
    previous_good_tracks = []
    while len(previous_good_tracks) != len(good_tracks):
        track_info['is_first_child_good'] = track_info['first_child_track_id'].isin(good_tracks)
        track_info['is_second_child_good'] = track_info['second_child_track_id'].isin(good_tracks)
        previous_good_tracks = good_tracks
        good_tracks = track_info[
            (track_info['track_length'].ge(min_track_length) & ~track_info['front_direction'].eq(0))
            | track_info['is_first_child_good']
            | track_info['is_second_child_good']
            # | track_info['track_end_position'].gt(channel_length - CHANNEL_END_TOLERANCE)
            ].index.unique()
    
    track_info['is_good_track'] = track_info.index.isin(good_tracks)


    track_info_with_children = track_info.join(track_info, on='first_child_track_id', rsuffix='_first_child').join(track_info, on='second_child_track_id', rsuffix='_second_child')
    
    while True:
        problematic_parents = track_info[
            track_info['is_good_track']
            & ~track_info['first_child_track_id'].isna()
            & ~track_info['is_first_child_good']
            & (
                track_info_with_children['track_end_position_first_child'].ge(channel_length - channel_end_tolerance)
                | track_info_with_children['track_end_position_first_child'].le(channel_end_tolerance)
            )
            & ~(
                track_info['is_second_child_good']
                & track_info['front_direction'].eq(track_info_with_children['front_direction_second_child'])
            )
            ].index
        if not len(problematic_parents):
            break
        problematic_children = track_info['first_child_track_id'][problematic_parents]
        track_info.loc[problematic_parents, 'is_first_child_good'] = True
        track_info.loc[problematic_children, 'is_good_track'] = True
        track_info.loc[problematic_children, 'front_direction'] = track_info.loc[problematic_parents, 'front_direction'].to_numpy()


    significant_tracks = track_info.index.unique()
    previous_significant_tracks = []
    while len(previous_significant_tracks) != len(significant_tracks):
        track_info['is_first_child_significant'] = track_info['first_child_track_id'].isin(significant_tracks)
        track_info['is_second_child_significant'] = track_info['second_child_track_id'].isin(significant_tracks)
        previous_significant_tracks = significant_tracks
        significant_tracks = track_info[
            (
                (track_info['track_length'] >= min_significant_track_length) 
                & (track_info['front_direction'] != 0)
            )
            | track_info['is_first_child_significant']
            | track_info['is_second_child_significant']
            # | track_info['track_end_position'].gt(channel_length - CHANNEL_END_TOLERANCE)
            ].index.unique()
    track_info['is_significant'] = track_info.index.isin(significant_tracks)
    track_info['splits_significantly'] = track_info['is_significant'] & track_info['is_first_child_significant'] & track_info['is_second_child_significant']

    return track_info


def get_front_fates(tracks, track_info, channel_length, v, channel_end_tolerance=8, te_back=30, te_forward=5, te_space=5, ending_search_radius=15):
    fates = track_info[
        track_info['is_good_track']
        # & track_info['front_direction'].eq(1)
        & ~track_info['is_first_child_good']
        & ~track_info['is_second_child_good']
        ].copy()

    if len(fates):
        has_near_neighbor = pd.Series(
        [(
            tracks['track_id'].ne(track_id) 
            & tracks['track_id'].ne(track_info.loc[track_id]['first_child_track_id'])
            & tracks['track_id'].ne(track_info.loc[track_id]['second_child_track_id'])
            & tracks['track_id'].ne(track_info.loc[track_id]['parent_track_id'])
            & tracks['track_id'].ne(track_info.loc[track_id]['sibling_track_id'])
            & tracks['seconds'].between(row['track_end'] - te_back, row['track_end'] + te_forward)
            & tracks['h'].between(row['track_end_position'] - te_space, row['track_end_position'] + te_space)
            ).any()
            for track_id, row in fates.iterrows()
        ], index=fates.index)

        nbrs = NearestNeighbors(n_neighbors=3)
        nbrs.fit(list(zip(fates['track_end'] * v, fates['track_end_position'])))
        nearest_endings = nbrs.radius_neighbors(radius=ending_search_radius, return_distance=False)
        front_directions = fates['front_direction'].to_numpy()
        has_near_ending = pd.Series([
            any(direction * front_directions[other_ending] < 1 for other_ending in nes) 
            for direction, nes in zip(front_directions, nearest_endings)
            ], index=fates.index) 
        
        fates['fate'] = 'failure' # note that this can be overriden
        fates['fate'] = fates['fate'].mask(has_near_neighbor | has_near_ending, 'anihilated') # note that this can be overriden
        fates['fate'] = fates['fate'].mask(fates['track_end_position'].ge(channel_length - channel_end_tolerance) & fates['front_direction'].eq(1), 'transmitted')
        fates['fate'] = fates['fate'].mask(fates['track_end_position'].lt(channel_end_tolerance) & fates['front_direction'].eq(-1), 'transmitted')
    else: # just to ensure that the column 'fate' is present in the output even if neither front is complete
        fates['fate'] = 'failure'

    return fates.sort_values(['tree_id', 'track_end'])


def get_significant_splits(track_info):
    significant_split_tracks = track_info[
        track_info['is_significant']
        & track_info['is_first_child_significant']
        & track_info['is_second_child_significant']
    ]

    significant_splits = pd.DataFrame({
        'parent_track_id': significant_split_tracks.index,
        'first_child_track_id': significant_split_tracks['first_child_track_id'],
        'second_child_track_id': significant_split_tracks['second_child_track_id'],
        'tree_id': significant_split_tracks['tree_id'],
        'significant_split_time': significant_split_tracks['track_end'],
        'significant_split_position': significant_split_tracks['track_end_position'],
    })

    return significant_splits


def get_input_pulse_to_tree_id(tracks, pulse_times):

    def get_nearest_peak_tree_id(input_pulse):
        near_maxima = tracks[tracks['seconds'].eq(input_pulse) & tracks['h'].le(3)]
        if len(near_maxima) == 1:
            return int(near_maxima.iloc[0]['tree_id'])
        else:
            print(f"WARNING: {len(near_maxima)} activity peaks found near channel start at time {input_pulse}.")
            return np.nan

    input_pulse_to_tree_id = [get_nearest_peak_tree_id(input_pulse) for input_pulse in pulse_times]

    previous_tree_id = -1
    for i in range(len(input_pulse_to_tree_id)):
        if input_pulse_to_tree_id[i] == previous_tree_id:
            input_pulse_to_tree_id[i] = np.nan
        else:
            previous_tree_id = input_pulse_to_tree_id[i]
    return input_pulse_to_tree_id



def get_pulse_fates(front_fates: pd.DataFrame, input_pulse_to_tree_id, significant_splits: pd.DataFrame, v, channel_length, channel_end_tolerance=8):
    front_fates = front_fates.assign(timespace= lambda x: x['track_end'] - x['track_end_position'] / v).sort_values(['tree_id', 'timespace'])

    pulse_fates = pd.DataFrame(
        [
            ({
                'fate': 'lost_somewhere',
                'track_end': np.nan,
                'track_end_position': np.nan,
             } if (arrivals['front_direction'] == 1).sum() == 0
            else arrivals[arrivals['front_direction'] == 1].iloc[0][['fate', 'track_end', 'track_end_position']].to_dict()
            )
            for tree_id in input_pulse_to_tree_id
            for arrivals in [
                front_fates[front_fates['tree_id'] == tree_id]
                ]
            ], index=input_pulse_to_tree_id
        )

    spawning_counts = pd.DataFrame([
        {
            'forward': (arrivals['front_direction'] == 1).sum() - 1,
            'short': (arrivals['front_direction'] == 0).sum(),
            'backward': (arrivals['front_direction'] == -1).sum(),
            'reached_end': (arrivals['track_end_position'].gt(channel_length - channel_end_tolerance)).sum(),
            'reached_start': (arrivals['track_end_position'].le(channel_end_tolerance)).sum(),
        }
        for tree_id in input_pulse_to_tree_id
        for arrivals in [
            front_fates[front_fates['tree_id'] == tree_id]
            ]
    ], index=input_pulse_to_tree_id)

    first_significant_splits = significant_splits.drop_duplicates('tree_id', keep='first').set_index('tree_id')[['significant_split_time', 'significant_split_position']]
    first_significant_splits = first_significant_splits.reindex(input_pulse_to_tree_id)
    first_significant_splits['pulse_id'] = list(range(len(input_pulse_to_tree_id)))

    pulse_fates_and_spawned = pd.concat([pulse_fates, spawning_counts, first_significant_splits], axis='columns')

    return pulse_fates_and_spawned


def plot_tracks(tracks, track_info, fates, pulse_fates=None, pulse_times=None, outpath=None, show=True, panel_size=(8,4)):
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

    if pulse_fates is not None and pulse_times is not None:
        axs[0].scatter(pulse_times, [0]*len(pulse_times), marker='^', c=[fate_to_color[fate] for fate in pulse_fates['fate']])
        axs[1].scatter(pulse_times, [0]*len(pulse_times), marker='^', c=[fate_to_color[fate] for fate in pulse_fates['fate']])

    if outpath is not None:
        plt.savefig(outpath)

    if show:
        plt.show()

    return fig, axs



def plot_kymograph_with_endings(activity, fates, duration, pulse_fates=None, pulse_times=None, significant_splits=None, outpath=None, show=True, panel_size=(10, 4)):

    fig, ax = plot_result_from_activity(activity, show=False, panel_size=panel_size)

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

    
    if pulse_fates is not None and pulse_times is not None:
        ax.scatter(np.array(pulse_times) / duration, [0]*len(pulse_times), marker='^', c=[fate_to_color[fate] for fate in pulse_fates['fate']])
   
    if significant_splits is not None and len(significant_splits):
        ax.scatter(significant_splits['significant_split_time'] / duration, significant_splits['significant_split_position'] , marker='o', edgecolors='cyan', facecolor='none')
   

    if outpath is not None:
        plt.savefig(outpath)

    if show:
        plt.show()


def plot_tracks_from_file(outdir, indir=None):
    outdir = Path(outdir)
    if indir is None:
        indir = outdir
    tracks = pd.read_csv(indir / 'tracks.csv')
    track_info = pd.read_csv(indir / 'track_info.csv').set_index('track_id')
    front_fates = pd.read_csv(indir / 'front_fates.csv').set_index('track_id')
    pulse_fates = pd.read_csv(indir / 'pulse_fates.csv')
    
    with open(indir / 'input_protocol.json') as file:
        input_protocol = json.load(file)
    pulse_times = [0] + list(np.cumsum(input_protocol))[:-1]

    panel_size = (tracks['frame'].max() / 100, (tracks['h'].max() + 1) / 100)
    return plot_tracks(tracks, track_info, front_fates, pulse_fates, pulse_times, show=False, outpath=(outdir / 'tracks.svg') if outdir else None, panel_size=panel_size)


def plot_kymograph_from_file(outdir, indir=None):
    outdir = Path(outdir)
    if indir is None:
        indir = outdir

    activity = pd.read_csv(indir / 'activity.csv')
    duration = pd.Series(activity.index.get_level_values('seconds').unique().sort_values()).diff()[0]
    front_fates = pd.read_csv(indir / 'front_fates.csv').set_index('track_id')
    pulse_fates = pd.read_csv(indir / 'pulse_fates.csv')
    
    with open(indir / 'input_protocol.json') as file:
        input_protocol = json.load(file)
    pulse_times = [0] + list(np.cumsum(input_protocol))[:-1]

    panel_size = (len(activity) / 100, (int(activity.columns[-1]) + 1) / 100)
    return plot_kymograph_with_endings(activity, front_fates, duration, pulse_fates, pulse_times, show=False, outpath=(outdir / 'kymograph.png') if outdir else None, panel_size=panel_size)



## ---------

def determine_fates(activity: pd.DataFrame = None, input_protocol: Iterable[float] = None, v=1/3.6, duration=5, channel_length=None, 
    min_distance_between_peaks=6, smoothing_sigma_h=1.5, smoothing_sigma_frames=1.2, min_peak_height=0.002, exp_mean=0.8, # parameters for "get_pulse_positions"
    laptrack_parameters={},
    front_direction_minimal_distance=5, min_track_length=5, min_significant_track_length=10, # parameters for "get_track_info"
    channel_end_tolerance=8, te_back=30, te_forward=5, te_space=6, ending_search_radius=15, # parameters for "get_front_fates" [channel_end_tolerance also used by "get_pulse_fates"]
    outdir=None, indir=None,
    plot_results=False, verbose=True, save_csv=True, use_cached=[],
    returns=['pulse_fates']):

    if outdir is not None and (save_csv or plot_results):
        outdir = Path(outdir)
        outdir.absolute().mkdir(exist_ok=True, parents=True)
    
    indir = indir or outdir


    # duration = pd.Series(activity.index.get_level_values('seconds')).diff().dropna().unique()[0]
    channel_length = channel_length or int(activity.columns[-1]) + 1
    pulse_times = [0] + list(np.cumsum(input_protocol))[:-1]

    if 'pulse_positions' in use_cached:
        pulse_positions = pd.read_csv(indir / 'pulse_positions.csv', index_col=0)
    else:
        if verbose: print('Determining pulse positions...')
        pulse_positions = get_pulse_positions(activity, min_distance_between_peaks=min_distance_between_peaks, smoothing_sigma_h=smoothing_sigma_h, smoothing_sigma_frames=smoothing_sigma_frames, min_peak_height=min_peak_height)
    if save_csv and outdir is not None:
        pulse_positions.to_csv(outdir / 'pulse_positions.csv')

    if verbose: print('Tracking...')
    tracks, split_df, merge_df = get_tracks(pulse_positions, duration, laptrack_parameters)
    if len(split_df) == 0:
        split_df = pd.DataFrame(columns=['parent_track_id', 'child_track_id'])
 
    if verbose: print('Extracting track information...')
    # This modifies tracks inplace, thus they are saved later!
    track_info = get_track_info(tracks, split_df, pulse_times, channel_length, v, front_direction_minimal_distance=front_direction_minimal_distance, min_track_length=min_track_length, min_significant_track_length=min_significant_track_length, channel_end_tolerance=channel_end_tolerance)
    if save_csv and outdir is not None:
        track_info.to_csv(outdir / 'track_info.csv')

    if save_csv and outdir is not None:
        tracks.to_csv(outdir / 'tracks.csv')


    if verbose: print('Determining front fates...')
    front_fates = get_front_fates(tracks, track_info, channel_length, v, channel_end_tolerance=channel_end_tolerance, te_back=te_back, te_forward=te_forward, te_space=te_space, ending_search_radius=ending_search_radius)
    if save_csv and outdir is not None:
        front_fates.to_csv(outdir / 'front_fates.csv')

    if verbose: print('Determining significant splits...')
    significant_splits = get_significant_splits(track_info)
    if save_csv and outdir is not None:
        significant_splits.to_csv(outdir / 'significant_splits.csv')

    if verbose: print('Matching input pulses with trees...')
    input_pulse_to_tree_id = get_input_pulse_to_tree_id(tracks, pulse_times)
    if save_csv and outdir is not None:
        pd.Series(input_pulse_to_tree_id).to_csv(outdir / 'input_pulse_to_tree_id.csv')
    
    if verbose: print('Determining pulse fates...')
    pulse_fates = get_pulse_fates(front_fates, input_pulse_to_tree_id, significant_splits, v, channel_length=channel_length, channel_end_tolerance=channel_end_tolerance)
    if save_csv and outdir is not None:
        pulse_fates.to_csv(outdir / 'pulse_fates.csv')

    
    if plot_results:
        if verbose: print("Plotting tracks...")
        panel_size = (tracks['frame'].max() / 100, (tracks['h'].max() + 1) / 100)
        plot_tracks(tracks, track_info, front_fates, pulse_fates, pulse_times, show=False, outpath=(outdir / 'out.svg') if outdir else None, panel_size=panel_size)
        plt.close()
        plot_kymograph_with_endings(activity, front_fates, duration, pulse_fates,  pulse_times, significant_splits=significant_splits, show=False, outpath=(outdir / 'out-kymo.png') if outdir else None, panel_size=panel_size)
        plt.close()

    if verbose: print('Done.')


    if len(returns) == 0:
        return None

    ret_dict = {
        'pulse_positions': pulse_positions,
        'tracks': tracks,
        'split_df': split_df,
        'merge_df': merge_df,
        'track_info': track_info,
        'front_fates': front_fates,
        'significant_splits': significant_splits,
        'input_pulse_to_tree_id': input_pulse_to_tree_id,
        'pulse_fates': pulse_fates,
    }
    ret_val = tuple(ret_dict[to_return] for to_return in returns)
    if len(returns) == 1:
        ret_val = ret_val[0]
    return ret_val


if __name__ == '__main__':
    indir = Path('../private/current_simulation')
    outdir = indir
    activity = pd.read_csv(indir / 'activity.csv')
    with open(indir / 'input_protocol.json') as file:
        input_protocol = json.load(file)
    
    determine_fates(activity, input_protocol, outdir=outdir, plot_results=True)


