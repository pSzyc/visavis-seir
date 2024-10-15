# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
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

from functools import cached_property

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


DEFAULT_LAPTRACK_PARAMETERS = {
    'track_dist_metric': "sqeuclidean",
    'splitting_dist_metric': "sqeuclidean",
    'merging_dist_metric': "sqeuclidean",

    'track_cost_cutoff': 5**2,
    'splitting_cost_cutoff': 14**2,
    'merging_cost_cutoff': False,  # no merging
    'gap_closing_max_frame_count': 0,
    'gap_closing_cost_cutoff': 0,

    'segment_start_cost': 3000,#10**2,
    'segment_end_cost': 3000,#10**2,
    'track_start_cost': 300,#20**2,
    'track_end_cost': 300,#20**2,

    'no_splitting_cost': 0,
    'no_merging_cost': 0,
}



class TrackedResults():
    """
    This class accepts 'activity' from SimulationResults,
    performs front detection and tracking, and determies pulse fates.
    """


    def __init__(
        self,
        activity: pd.DataFrame = None, input_protocol: Iterable[float] = None, v=1/3.6, logging_interval=5, channel_length=None, 
        min_distance_between_peaks=6, smoothing_sigma_h=1.5, smoothing_sigma_frames=1.2, min_peak_height=0.002, smoothing_exp_frames=0.8, # parameters for "get_pulse_positions"
        laptrack_parameters={},
        front_direction_minimal_distance=5, min_track_length=5, min_significant_track_length=10, # parameters for "get_track_info"
        channel_end_tolerance=8, te_back=30, te_forward=5, te_space=6, ending_search_radius=15, # parameters for "get_front_fates" [channel_end_tolerance also used by "get_pulse_fates"]
        maximal_distance_in_event=20, maximal_interval_in_event=150, # parameters for "get_hierarchy"
        outdir=None, indir=None,
        plot_results=False, verbose=True, save_csv=True, use_cached=False,
        lazy=True,
    ):
        self.activity = activity
        self.input_protocol = input_protocol,
        self.pulse_times = [0] + list(np.cumsum(input_protocol))[:-1]
        self.v = v
        self.logging_interval = logging_interval,
        self.channel_length = channel_length or int(activity.columns[-1]) + 1
        self.min_distance_between_peaks = min_distance_between_peaks
        self.smoothing_sigma_h = smoothing_sigma_h
        self.smoothing_sigma_frames = smoothing_sigma_frames
        self.min_peak_height = min_peak_height
        self.smoothing_exp_frames = smoothing_exp_frames
        self.laptrack_parameters = {**DEFAULT_LAPTRACK_PARAMETERS, **laptrack_parameters}
        self.front_direction_minimal_distance = front_direction_minimal_distance
        self.min_track_length = min_track_length
        self.min_significant_track_length = min_significant_track_length
        self.channel_end_tolerance = channel_end_tolerance
        self.te_back = te_back
        self.te_forward = te_forward
        self.te_space = te_space
        self.ending_search_radius = ending_search_radius
        self.maximal_distance_in_event = maximal_distance_in_event
        self.maximal_interval_in_event = maximal_interval_in_event
        self.outdir = outdir
        self.indir = indir or outdir
        self.plot_results = plot_results
        self.verbose = verbose
        self.save_csv = save_csv
        self.use_cached = use_cached


        if not lazy:
            self.pulse_positions
            self.tracks
            self.splits
            self.merges
            self.track_info
            self.front_fates
            self.significant_splits
            self.input_pulse_to_tree_id
            self.pulse_fates
            self.split_events


        if plot_results:
            if self.verbose: print("Plotting tracks...")
            panel_size = (self.tracks['frame'].max() / 100, (self.tracks['h'].max() + 1) / 100)
            self.plot_tracks(
                # tracks, track_info, front_fates, pulse_fates, pulse_times, show=False, outpath=(outdir / 'out.svg') if outdir else None,
                panel_size=panel_size, show=False)
            plt.close()
            self.plot_kymograph_with_endings(
                # activity, front_fates, logging_interval, pulse_fates,  pulse_times, significant_splits=significant_splits, show=False, outpath=(outdir / 'out-kymo.png') if outdir else None, 
                panel_size=panel_size, show=False)
            plt.close()


    @property
    def pulse_positions(self):
        return self._read_from_cache_or_compute('pulse_positions', message='Determining pulse positions...')

    @property
    def tracks(self):
        private_property = '_tracks'
        if not hasattr(self, private_property) or getattr(self, private_property) is None:
            cache_file = self.outdir / 'tracks.csv'
            if self.use_cached and cache_file.exists():
                self._tracks = pd.read_csv(cache_file).set_index('Unnamed: 0')
            else:
                self._compute_tracks_and_info()
        return self._tracks
        
    @property
    def splits(self):
        private_property = '_splits'
        if not hasattr(self, private_property) or getattr(self, private_property) is None:
            cache_file = self.outdir / 'splits.csv'
            if self.use_cached and cache_file.exists():
                self._splits = pd.read_csv(cache_file).set_index('Unnamed: 0')
            else:
                self._compute_tracks_and_info()
        return self._splits
        
    @property
    def merges(self):
        private_property = '_merges'
        if not hasattr(self, private_property) or getattr(self, private_property) is None:
            cache_file = self.outdir / 'merges.csv'
            if self.use_cached and cache_file.exists():
                self._merges = pd.read_csv(cache_file).set_index('Unnamed: 0')
            else:
                self._compute_tracks_and_info()
        return self._merges
    
    @property
    def track_info(self):
        private_property = '_track_info'
        if not hasattr(self, private_property) or getattr(self, private_property) is None:
            cache_file = self.outdir / 'track_info.csv'
            if self.use_cached and cache_file.exists():
                self._track_info = pd.read_csv(cache_file).set_index('track_id')
            else:
                self._compute_tracks_and_info()
        return self._track_info

    @property
    def front_fates(self):
        return self._read_from_cache_or_compute('front_fates', message='Determining front fates...', index=['track_id'])

    @property
    def significant_splits(self):
        return self._read_from_cache_or_compute('significant_splits', message='Determining significant splits...', index=['track_id'])

    @property
    def input_pulse_to_tree_id(self):
        return self._read_from_cache_or_compute('input_pulse_to_tree_id', message='Matching input pulses with trees...', index=['input_pulse'])

    @property
    def pulse_fates(self):
        return self._read_from_cache_or_compute('pulse_fates', message='Determining pulse fates...', index=['tree_id'])

    @property
    def significant_split_hierarchy(self):
        return self._read_from_cache_or_compute('significant_split_hierarchy', message='Computing the split tree...', index=['Unnamed: 0'])

    @property
    def effective_front_directions(self):
        return self._read_from_cache_or_compute('effective_front_directions', message='Computing effective front directions...', index=['track_id'])
        
    @property
    def split_events(self):
        return self._read_from_cache_or_compute('split_events', message='Grouping splits into indepenent events...', index=['event_id'])


    def _read_from_cache_or_compute(self, property, message='', index=None):
        '''Checks if self._{property} exists.
        If not, tries loading from outpath / '{property}.csv'.
        If the cache file does not exist or use_cache=False,
            computes the value and stores it to self._{property} and to csv.
        '''
        private_property = f'_{property}'
        value = getattr(self, private_property) if hasattr(self, private_property) else None
        if value is None:
            cache_file = self.outdir / f'{property}.csv'
            if self.use_cached and cache_file.exists():
                value = pd.read_csv(cache_file)
                if index:
                    value = value.set_index(index)
            else:
                if self.verbose: print(message, end=' ')
                # print(property, sorted(self.__dict__.keys()))
                value = getattr(self, f'_get_{property}')()
                if self.save_csv: value.to_csv(cache_file)
                if self.verbose: print('done.')
            setattr(self, private_property, value)
                
        return value


    def _get_pulse_positions(self):

        exp_kernel_size = int(3 * self.smoothing_exp_frames) + 1
        exp_kernel = np.exp(-np.arange(0, exp_kernel_size) / self.smoothing_exp_frames)
        exp_kernel = exp_kernel / exp_kernel.sum()

        smoothed_data = convolve1d(
            gaussian_filter1d(
                1.*self.activity.to_numpy(), sigma=self.smoothing_sigma_h, axis=1, mode='constant', truncate=2.01
            ), # smoothing along spatial axis
            exp_kernel, # smoothing along temporal axis
            origin=-(exp_kernel_size // 2),
            mode='nearest',
            axis=0,
        )

        pulse_positions = []
        for seconds, activity_slice in zip(self.activity.index.get_level_values('seconds'), smoothed_data):
            pulse_positions_part, _ = find_peaks(activity_slice, distance=self.min_distance_between_peaks, height=self.min_peak_height)
            pulse_positions.extend(
                {
                    'seconds': seconds,
                    'h': h,
                } for h in [-200] + list(pulse_positions_part) # -200 is an artificial maximum added so that LapTrack does not omit emtpy frames
            )
        
        pulse_positions = pd.DataFrame(pulse_positions)
        return pulse_positions


    def _compute_tracks_and_info(self):
        if self.verbose: print('Tracking...', end=' ')
        self._compute_tracks()
        if self.verbose: print('done.')

        if self.verbose: print('Extracting track information...', end=' ')
        self._compute_track_info()
        if self.verbose: print('done.')

        if self.save_csv:
            self._tracks.to_csv(self.outdir / 'tracks.csv')
            self._splits.to_csv(self.outdir / 'splits.csv')
            self._merges.to_csv(self.outdir / 'merges.csv')
            self._track_info.to_csv(self.outdir / 'track_info.csv')


    # Tracks must be post-processed by self_.compute_track_info after running this fuction to be complete !!!
    def _compute_tracks(self):

        lt = LapTrack(**self.laptrack_parameters)

        self.pulse_positions['frame'] = self.pulse_positions['seconds'] // self.logging_interval

        track_df, split_df, merge_df = lt.predict_dataframe(
            self.pulse_positions,
            coordinate_cols=['h'],
            frame_col='frame',
            only_coordinate_cols=False,
        )
        tracks = track_df.reset_index(drop=True).rename(columns={'frame_y': 'frame'})
        tracks = tracks[tracks['h'] != -200] # remove the artificial track
        
        if len(split_df) == 0:
            split_df = pd.DataFrame(columns=['parent_track_id', 'child_track_id'])
        
        if len(merge_df) == 0:
            merge_df = pd.DataFrame(columns=['parent_track_id', 'child_track_id'])
 
        self._tracks = tracks
        self._splits = split_df
        self._merges = merge_df
    

    # This modifies self._tracks inplace !!!
    def _compute_track_info(self):
        
        assert hasattr(self, '_tracks')
        assert hasattr(self, '_splits')


        track_info = (self._tracks.groupby('track_id')
                    ['seconds'].agg(['size', 'min', 'max']).rename(columns={'size': 'track_length', 'min': 'track_start', 'max': 'track_end'})
                    .join(self._tracks[['track_id', 'tree_id']].drop_duplicates().set_index('track_id'), how='left')
        )
        track_info.index.name = 'track_id'

        track_info['track_start_position'] = self._tracks.set_index(['track_id', 'seconds']).reindex(
            list(zip(track_info.index, track_info['track_start']))
            ).reset_index('seconds')['h']
        track_info['track_end_position'] = self._tracks.set_index(['track_id', 'seconds']).reindex(
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
            self._tracks['track_id'].replace(child_track_id, parent_track_id, inplace=True)
            track_info.loc[parent_track_id, ['track_end', 'track_end_position']] = track_info.loc[child_track_id, ['track_end', 'track_end_position']]
            track_info.loc[parent_track_id, 'track_length'] = track_info.loc[parent_track_id, 'track_length'] + track_info.loc[child_track_id]['track_length']
            track_info.drop(index=child_track_id, inplace=True)
            if child_track_id in enhanced_split_df.index:
                enhanced_split_df.loc[parent_track_id] = enhanced_split_df.loc[child_track_id]
                enhanced_split_df.drop(index=child_track_id, inplace=True)
            else:
                enhanced_split_df.drop(index=parent_track_id, inplace=True)

        n_trees = track_info['tree_id'].max() + 1
        
        enhanced_split_df = self._splits.copy()
        enhanced_split_df['child_no'] = np.arange(len(self._splits)) % 2
        # print(self.outdir, enhanced_split_df)
        enhanced_split_df = enhanced_split_df.set_index(['parent_track_id', 'child_no'])['child_track_id'].unstack('child_no').rename(columns={0: 'first_child_track_id', 1: 'second_child_track_id'}).reindex(columns=['first_child_track_id', 'second_child_track_id'])
        

        track_info_with_children = (
            track_info
                .join(enhanced_split_df)
                .join(track_info, on='first_child_track_id', rsuffix='_first_child')
                .join(track_info, on='second_child_track_id', rsuffix='_second_child')
        )

        tracks_with_first_child_at_pulse = track_info_with_children[
            track_info_with_children['track_start_position_first_child'].le(3)
            & track_info_with_children['track_start_first_child'].isin(self.pulse_times)
        ]

        for parent_track_id, track in tracks_with_first_child_at_pulse.iterrows():
            track_id = track['first_child_track_id']
            sibling_track_id = track['second_child_track_id']
            merge_tracks(parent_track_id, sibling_track_id)
            assign_tree_id_recursively(track_id, n_trees)
            n_trees += 1

        tracks_with_second_child_at_pulse = track_info_with_children[
            track_info_with_children['track_start_position_second_child'].le(3)
            & track_info_with_children['track_start_second_child'].isin(self.pulse_times)
        ]

        for parent_track_id, track in tracks_with_second_child_at_pulse.iterrows():
            track_id = track['second_child_track_id']
            sibling_track_id = track['first_child_track_id']
            merge_tracks(parent_track_id, sibling_track_id)
            assign_tree_id_recursively(track_id, n_trees)
            n_trees += 1

        self._tracks['tree_id'] = self._tracks.join(track_info, on='track_id', lsuffix='_old')['tree_id']
        track_info = track_info.join(enhanced_split_df)
        first_children = track_info['first_child_track_id'].dropna()
        second_children = track_info['second_child_track_id'].dropna()
        track_info.loc[first_children, 'parent_track_id'] = first_children.index
        track_info.loc[first_children, 'sibling_track_id'] = second_children.to_numpy()
        track_info.loc[second_children, 'parent_track_id'] = second_children.index
        track_info.loc[second_children, 'sibling_track_id'] = first_children.to_numpy()


        track_info['front_direction'] = (
            1 *   (  track_info['track_end_position'] - track_info['track_start_position']).ge(
                np.minimum(self.front_direction_minimal_distance, 0.5 * self.v * (track_info['track_end'] - track_info['track_start']))) 
            - 1 * (-(track_info['track_end_position'] - track_info['track_start_position'])).ge(
                np.minimum(self.front_direction_minimal_distance, 0.5 * self.v * (track_info['track_end'] - track_info['track_start']))) 
        )


        good_tracks = track_info.index.unique()
        previous_good_tracks = []
        while len(previous_good_tracks) != len(good_tracks):
            track_info['is_first_child_good'] = track_info['first_child_track_id'].isin(good_tracks)
            track_info['is_second_child_good'] = track_info['second_child_track_id'].isin(good_tracks)
            previous_good_tracks = good_tracks
            good_tracks = track_info[
                (track_info['track_length'].ge(self.min_track_length) & ~track_info['front_direction'].eq(0))
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
                    track_info_with_children['track_end_position_first_child'].ge(self.channel_length - self.channel_end_tolerance)
                    | track_info_with_children['track_end_position_first_child'].le(self.channel_end_tolerance)
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
                    (track_info['track_length'] >= self.min_significant_track_length) 
                    & (track_info['front_direction'] != 0)
                )
                | track_info['is_first_child_significant']
                | track_info['is_second_child_significant']
                # | track_info['track_end_position'].gt(channel_length - CHANNEL_END_TOLERANCE)
                ].index.unique()
        track_info['is_significant'] = track_info.index.isin(significant_tracks)
        track_info['splits_significantly'] = track_info['is_significant'] & track_info['is_first_child_significant'] & track_info['is_second_child_significant']

        self._track_info = track_info


    def _get_front_fates(self):
        fates = self.track_info[
            self.track_info['is_good_track']
            # & track_info['front_direction'].eq(1)
            & ~self.track_info['is_first_child_good']
            & ~self.track_info['is_second_child_good']
            ].copy()

        if len(fates):
            has_near_neighbor = pd.Series(
            [(
                self.tracks['track_id'].ne(track_id) 
                & self.tracks['track_id'].ne(self.track_info.loc[track_id]['first_child_track_id'])
                & self.tracks['track_id'].ne(self.track_info.loc[track_id]['second_child_track_id'])
                & self.tracks['track_id'].ne(self.track_info.loc[track_id]['parent_track_id'])
                & self.tracks['track_id'].ne(self.track_info.loc[track_id]['sibling_track_id'])
                & self.tracks['seconds'].between(row['track_end'] - self.te_back, row['track_end'] + self.te_forward)
                & self.tracks['h'].between(row['track_end_position'] - self.te_space, row['track_end_position'] + self.te_space)
                ).any()
                for track_id, row in fates.iterrows()
            ], index=fates.index)

            nbrs = NearestNeighbors(n_neighbors=3)
            nbrs.fit(list(zip(fates['track_end'] * self.v, fates['track_end_position']))) 
            nearest_endings = nbrs.radius_neighbors(radius=self.ending_search_radius, return_distance=False)
            front_directions = fates['front_direction'].to_numpy()
            has_near_ending = pd.Series([
                any(direction * front_directions[other_ending] < 1 for other_ending in nes) 
                for direction, nes in zip(front_directions, nearest_endings)
                ], index=fates.index) 
            
            fates['fate'] = 'failure' # note that this can be overriden
            fates['fate'] = fates['fate'].mask(has_near_neighbor | has_near_ending, 'anihilated') # note that this can be overriden
            fates['fate'] = fates['fate'].mask(fates['track_end_position'].ge(self.channel_length - self.channel_end_tolerance) & fates['front_direction'].eq(1), 'transmitted')
            fates['fate'] = fates['fate'].mask(fates['track_end_position'].lt(self.channel_end_tolerance) & fates['front_direction'].eq(-1), 'transmitted')
        else: # just to ensure that the column 'fate' is present in the output even if neither front is complete
            fates['fate'] = 'failure'

        return fates.sort_values(['tree_id', 'track_end'])


    def _get_significant_splits(self):
        significant_split_tracks = self.track_info[
            self.track_info['is_significant']
            & self.track_info['is_first_child_significant']
            & self.track_info['is_second_child_significant']
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

    
    def _get_input_pulse_to_tree_id(self):

        def get_nearest_peak_tree_id(input_pulse):
            near_maxima = self.tracks[self.tracks['seconds'].eq(input_pulse) & self.tracks['h'].le(3)]
            if len(near_maxima) == 1:
                return int(near_maxima.iloc[0]['tree_id'])
            else:
                print(f"WARNING: {len(near_maxima)} activity peaks found near channel start at time {input_pulse}.")
                return np.nan

        input_pulse_to_tree_id = [get_nearest_peak_tree_id(input_pulse) for input_pulse in self.pulse_times]

        previous_tree_id = -1
        for i in range(len(input_pulse_to_tree_id)):
            if input_pulse_to_tree_id[i] == previous_tree_id:
                input_pulse_to_tree_id[i] = np.nan
            else:
                previous_tree_id = input_pulse_to_tree_id[i]
        input_pulse_to_tree_id_df =  pd.DataFrame({'tree_id': input_pulse_to_tree_id})
        input_pulse_to_tree_id_df.index.name = 'input_pulse'
        return input_pulse_to_tree_id_df


    def _get_pulse_fates(self):
        front_fates = (
            self.front_fates
            .assign(timespace= lambda x: x['track_end'] - x['track_end_position'] / self.v)
            .sort_values(['tree_id', 'timespace'])
        )

        pulse_fates = pd.DataFrame(
            [
                ({
                    'fate': 'lost_somewhere',
                    'track_end': np.nan,
                    'track_end_position': np.nan,
                } if (arrivals['front_direction'] == 1).sum() == 0
                else arrivals[arrivals['front_direction'] == 1].iloc[0][['fate', 'track_end', 'track_end_position']].to_dict()
                )
                for tree_id in self.input_pulse_to_tree_id['tree_id']
                for arrivals in [
                    front_fates[front_fates['tree_id'] == tree_id]
                    ]
                ], index=self.input_pulse_to_tree_id['tree_id']
            )

        spawning_counts = pd.DataFrame([
            {
                'forward': (arrivals['front_direction'] == 1).sum() - 1,
                'short': (arrivals['front_direction'] == 0).sum(),
                'backward': (arrivals['front_direction'] == -1).sum(),
                'reached_end': (arrivals['track_end_position'].gt(self.channel_length - self.channel_end_tolerance)).sum(),
                'reached_start': (arrivals['track_end_position'].le(self.channel_end_tolerance)).sum(),
            }
            for tree_id in self.input_pulse_to_tree_id['tree_id']
            for arrivals in [
                front_fates[front_fates['tree_id'] == tree_id]
                ]
        ], index=self.input_pulse_to_tree_id['tree_id'])

        first_significant_splits = self.significant_splits.drop_duplicates('tree_id', keep='first').set_index('tree_id')[['significant_split_time', 'significant_split_position']]
        first_significant_splits = first_significant_splits.reindex(self.input_pulse_to_tree_id['tree_id'])
        first_significant_splits['pulse_id'] = list(range(len(self.input_pulse_to_tree_id)))

        pulse_fates_and_spawned = pd.concat([pulse_fates, spawning_counts, first_significant_splits], axis='columns')

        return pulse_fates_and_spawned



    def _get_significant_split_hierarchy(self):
        hierarchy = pd.DataFrame(         
            index=self.significant_splits.index if len(self.significant_splits.index) else [],
            columns=['parent_significant_split', 'first_child_significant_split', 'second_child_significant_split', 'is_same_event_as_parent', 'event_id'],
            # dtype=float,
            )
        hierarchy['is_same_event_as_parent'] = False
        for track_id, row in self.significant_splits.iterrows():
            up_track_id = track_id
            while True:
                prev_track_id = up_track_id
                up_track_id = self.track_info.loc[up_track_id, 'parent_track_id']
                if np.isnan(up_track_id):
                    hierarchy.loc[track_id, 'parent_significant_split'] = np.nan
                    break
                if up_track_id in self.significant_splits.index:
                    hierarchy.loc[track_id, 'parent_significant_split'] = up_track_id
                    hierarchy.loc[track_id, 'is_same_event_as_parent'] = (
                            (np.abs(self.significant_splits.loc[up_track_id, 'significant_split_position'] - self.significant_splits.loc[track_id, 'significant_split_position']) < self.maximal_distance_in_event)
                            & (np.abs(self.significant_splits.loc[up_track_id, 'significant_split_time'] - self.significant_splits.loc[track_id, 'significant_split_time']) < self.maximal_interval_in_event)
                        )
                    if self.track_info.loc[up_track_id, 'first_child_track_id'] == prev_track_id:
                        hierarchy.loc[up_track_id, 'first_child_significant_split'] = track_id
                    else:
                        hierarchy.loc[up_track_id, 'second_child_significant_split'] = track_id
                    
                    break
        n_events = (~hierarchy['is_same_event_as_parent']).sum()
        hierarchy.loc[hierarchy[~hierarchy['is_same_event_as_parent']].index, 'event_id'] = list(range(n_events))

        to_update = hierarchy[hierarchy['is_same_event_as_parent']].index
        while hierarchy['event_id'].isna().any():
            hierarchy.loc[to_update, 'event_id'] = hierarchy.join(hierarchy, on='parent_significant_split', lsuffix='_child').loc[to_update, 'event_id']

        hierarchy['is_first_child_free'] = hierarchy.join(hierarchy, on='first_child_significant_split', rsuffix='_first_child').pipe(lambda x: ~(x['event_id_first_child'] == x['event_id']))
        hierarchy['is_second_child_free'] = hierarchy.join(hierarchy, on='second_child_significant_split', rsuffix='_second_child').pipe(lambda x: ~(x['event_id_second_child'] == x['event_id']))
            
        return hierarchy


    def _get_effective_front_directions(self):
        effective_front_direction = self.track_info['front_direction'] * self.track_info['is_significant'] * (self.track_info['track_length'] >= 40/5)
        effective_parent = pd.Series(self.track_info.index, index=self.track_info.index, name='effective_parent')

        child_track_info = self.track_info
        while len(child_track_info):
            the_child = child_track_info['first_child_track_id'].where(child_track_info['is_first_child_significant'] & ~child_track_info['is_second_child_significant'],
                        child_track_info['second_child_track_id'].where(~child_track_info['is_first_child_significant'] & child_track_info['is_second_child_significant'],
                        np.nan
                    )).dropna()
            
            the_effective_parent = the_child.index
            effective_parent[the_child] = the_effective_parent

            child_track_info = self.track_info.loc[the_child]
            child_track_info.index = the_effective_parent
            child_front_direction = effective_front_direction.loc[the_child]
            child_front_direction.index = the_effective_parent
            parent_front_direction = effective_front_direction.loc[the_effective_parent]
            effective_front_direction.loc[the_effective_parent] = (
                parent_front_direction.where(
                    child_front_direction.eq(0) | child_front_direction.eq(parent_front_direction), 
                child_front_direction.where(
                    parent_front_direction.eq(0),
                np.nan))
            )


        effective_front_direction.name = 'effective_front_direction'

        effective_front_direction = effective_front_direction.fillna(0)
        effective_front_direction = effective_parent.to_frame().join(effective_front_direction, on='effective_parent')['effective_front_direction'].astype(int)
        
        return effective_front_direction
            
            

    def _get_split_events(self):
        significant_splits_with_hierarchy = self.significant_splits.join(self.significant_split_hierarchy).join(self.effective_front_directions)
        event_ids = significant_splits_with_hierarchy['event_id'].unique().astype(int)
        events = pd.DataFrame(index=event_ids, columns=['parent_track_id', 'parent_front_direction', 'forward', 'uncertain', 'backward', 'absolutely_forward', 'absolutely_backward'])
        events.index.name = 'event_id'
        for event_id, splits in significant_splits_with_hierarchy.groupby('event_id'):
            parent_track_id = splits[~splits['is_same_event_as_parent']].index[0]
            parent_front_direction = splits.loc[parent_track_id, 'effective_front_direction']

            front_direction_counts = (
                splits[splits['is_first_child_free']].join(self.effective_front_directions, on='first_child_track_id', lsuffix='_parent').value_counts('effective_front_direction').reindex([1,0,-1], fill_value=0)
                + splits[splits['is_second_child_free']].join(self.effective_front_directions, on='second_child_track_id', lsuffix='_parent').value_counts('effective_front_direction').reindex([1,0,-1], fill_value=0)
            ) - pd.Series(1, index=[parent_front_direction]).reindex([1,0,-1], fill_value=0)


            events.loc[event_id, 'parent_track_id'] = parent_track_id
            events.loc[event_id, 'parent_front_direction'] = parent_front_direction
            events.loc[event_id, ['absolutely_forward', 'uncertain', 'absolutely_backward']] = front_direction_counts.to_numpy()
            events.loc[event_id, ['forward', 'backward']] = (
                front_direction_counts.loc[[1,-1]].to_numpy() if parent_front_direction == 1 
                else front_direction_counts.loc[[-1,1]].to_numpy() if parent_front_direction == -1
                else (front_direction_counts.loc[[1,-1]].to_numpy() + front_direction_counts.loc[[-1,1]].to_numpy())/2
            )
        return events




    # ----------- Plotting ---------------



    def plot_tracks(self, show=True, panel_size=(8,4), filename='tracks.svg'):
        fig, axs = subplots_from_axsize(2, 1, panel_size)

        good_tracks = self.track_info[self.track_info['is_good_track']].index

        tracks_selected = self.tracks[self.tracks['track_id'].isin(good_tracks)]#[(tracks['track_id'] >= 0) & (tracks['track_id'] <= 31)]
        tracks_not_selected = self.tracks[~self.tracks['track_id'].isin(good_tracks)]#[(tracks['track_id'] >= 0) & (tracks['track_id'] <= 31)]


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
            events = self.front_fates[self.front_fates['fate'] == fate]
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

        if self.pulse_fates is not None and self.pulse_times is not None:
            axs[0].scatter(self.pulse_times, [0]*len(self.pulse_times), marker='^', c=[fate_to_color[fate] for fate in self.pulse_fates['fate']])
            axs[1].scatter(self.pulse_times, [0]*len(self.pulse_times), marker='^', c=[fate_to_color[fate] for fate in self.pulse_fates['fate']])

        if self.outdir is not None and filename is not None:
            plt.savefig(self.outdir / filename)

        if show:
            plt.show()

        return fig, axs



    def plot_kymograph_with_endings(self, show=True, panel_size=(10, 4), filename='kymograph.png'):

        fig, ax = plot_result_from_activity(self.activity, show=False, panel_size=panel_size)

        for fate, color in (
            ('transmitted', 'blue'),
            ('failure', 'green'),
            ('anihilated', 'red'),
        ):
            events = self.front_fates[self.front_fates['fate'] == fate]
            ax.scatter(
                events['track_end'] / self.logging_interval,
                events['track_end_position'],
                marker='x',
                color=color,
            )

        
        if self.pulse_fates is not None and self.pulse_times is not None:
            ax.scatter(np.array(self.pulse_times) / self.logging_interval, [0]*len(self.pulse_times), marker='^', c=[fate_to_color[fate] for fate in self.pulse_fates['fate']])
    
        if self.significant_splits is not None and len(self.significant_splits):
            ax.scatter(self.significant_splits['significant_split_time'] / self.logging_interval, self.significant_splits['significant_split_position'] , marker='o', edgecolors='cyan', facecolor='none')
    

        if self.outdir is not None and filename is not None:
            plt.savefig(self.outdir / filename)

        if show:
            plt.show()


    def plot_tracks_from_file(self, **kwargs):
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
        return self.plot_tracks(panel_size=panel_size, **kwargs)


    def plot_kymograph_from_file(self,
                                #  outdir, indir=None
                                **kwargs
                                 ):
        outdir = Path(outdir)
        if indir is None:
            indir = outdir

        activity = pd.read_csv(indir / 'activity.csv')
        logging_interval = pd.Series(activity.index.get_level_values('seconds').unique().sort_values()).diff()[0]
        front_fates = pd.read_csv(indir / 'front_fates.csv').set_index('track_id')
        pulse_fates = pd.read_csv(indir / 'pulse_fates.csv')
        
        with open(indir / 'input_protocol.json') as file:
            input_protocol = json.load(file)
        pulse_times = [0] + list(np.cumsum(input_protocol))[:-1]

        panel_size = (len(activity) / 100, (int(activity.columns[-1]) + 1) / 100)
        return self.plot_kymograph_with_endings(panel_size=panel_size, **kwargs)



    ## ---------

