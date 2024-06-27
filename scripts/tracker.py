# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
from bisect import bisect_left
from warnings import warn
import pandas as pd
import numpy as np



def make_tracks(
    peak_positions, 
    channel_length,
    duration=5,
    v=1/3.6,
    frame_field='seconds',
    max_distance_within_track = 4,
    start_region = 3,
    end_region_size = 7,
    max_split_distance = 9,
    max_merge_distance = 6,
):
    max_frames_for_split = int(max_split_distance / duration / v)

    end_region = channel_length - end_region_size
    n_tracks = 0
    n_frames = len(peak_positions)

    track_ids = []
    tracks = []
    # track_starts = []
    # track_ends = []
    
    def attach(frame, it_curr, it_prev):
        assert len(track_ids[frame]) == it_curr
        track_ids[frame].append(track_ids[frame - 1][it_prev])

    def mark_start(frame, peak_id, peak_position):
        nonlocal n_tracks
        assert len(track_ids[frame]) == peak_id
        track_ids[frame].append(n_tracks)
        # track_starts.append((n_tracks, frame, peak_id))
        tracks.append({
            'track_id': n_tracks,
            'start_frame':  frame,
            'start_position':  peak_position,
        })
        n_tracks += 1

    def mark_end(frame, peak_id, peak_position):
        track_id = track_ids[frame][peak_id]
        tracks[track_id].update({
            'end_frame': frame,
            'end_position': peak_position,
        })
        # track_ends.append((track_ids[frame][peak_id], frame, peak_id))

    prev = []

    for frame, curr in enumerate(peak_positions + [[]]):

        track_ids.append([])
        it_prev = 0
        it_curr = 0

        while True:
            if it_prev >= len(prev):
                for i in range(it_curr, len(curr)):
                    mark_start(frame, i, curr[i])
                break
            if it_curr >= len(curr):
                for i in range(it_prev, len(prev)):
                    mark_end(frame - 1, i, prev[i])
                break
            
            diff = curr[it_curr] - prev[it_prev]
            if diff > max_distance_within_track:
                mark_end(frame - 1, it_prev, prev[it_prev])
                it_prev += 1
                continue
            if -max_distance_within_track <= diff <= max_distance_within_track:
                if it_curr < len(curr) - 1 and (curr[it_curr + 1] - prev[it_prev])**2 < diff**2:
                    mark_start(frame, it_curr, curr[it_curr])
                    it_curr += 1
                    continue
                attach(frame, it_curr, it_prev)
                it_prev += 1
                it_curr += 1
                continue
            if diff < -max_distance_within_track:
                mark_start(frame, it_curr, curr[it_curr])
                it_curr += 1
                continue
            raise Exception('This code should not be reached')
        
        prev = curr


    ### evaluation of track starts


    n_trees = 0

    def mark_new_pulse(track_id):
        nonlocal n_trees
        tracks[track_id].update({
            'tree_id': n_trees,
            'source': 'start',
            'parent_track_id': None,
            'split_frame': None,
            })
        n_trees += 1

    def mark_end_pulse(track_id):
        nonlocal n_trees
        tracks[track_id].update({
            'tree_id': n_trees,
            'source': 'end',
            'parent_track_id': None,
            'split_frame': None,
            })
        n_trees += 1

    def mark_inherited(track_id):
        nonlocal n_trees
        tracks[track_id].update({
            'tree_id': n_trees,
            'source': 'leftover',
            'parent_track_id': None,
            'split_frame': None,
            })
        n_trees += 1

    def mark_split(parent_track_id, child_track_id, split_frame):
        tracks[child_track_id].update({
            'tree_id': tracks[parent_track_id]['tree_id'],
            'source': 'split',
            'parent_track_id': parent_track_id,
            'split_frame': split_frame,
            })
        
    def mark_emerge(track_id):
        nonlocal n_trees
        tracks[track_id].update({
            'tree_id': n_trees,
            'source': 'emerge',
            'parent_track_id': None,
            'split_frame': None,
            })
        n_trees += 1



    for track_id, track in enumerate(tracks):
        start_position = track['start_position']
        frame = track['start_frame']
        if start_position <= start_region:
            mark_new_pulse(track_id)
            continue
        if frame  == 0:
            mark_inherited(track_id)
            continue
        prev = peak_positions[frame - 1]

        candidates_for_nearest = [
            (fr, peak_id)
                for fr in range(max(frame - max_frames_for_split, 0), frame) 
                for peak_id in range(len(peak_positions[fr]))]

        if len(candidates_for_nearest):

            sq_distances = [(v * duration * (fr - frame)) ** 2 + (peak_positions[fr][peak_id] - start_position)**2 for fr, peak_id in candidates_for_nearest]
            nearest = np.argmin(sq_distances)
            print(nearest, sq_distances[nearest])
            if sq_distances[nearest] <= max_split_distance**2:
                nearest_frame, nearest_peak_id = candidates_for_nearest[nearest]
                mark_split(track_ids[nearest_frame][nearest_peak_id], track_id, nearest_frame)
                continue
        if start_position >= end_region:
            mark_end_pulse(track_id)
            continue
        mark_emerge(track_id)
        warn(f"Track {track_id} seems to emerge from nowhere at position {start_position} in frame {frame}")
    

    ### evaluation of track ends

    def mark_disappeared_at_start(track_id):
        tracks[track_id].update({
            'drain': 'start',
            'stem_track_id': None,
            })
    
    def mark_disappeared_at_end(track_id):
        tracks[track_id].update({
            'drain': 'end',
            'stem_track_id': None,
            })

    def mark_leftover(track_id):
        tracks[track_id].update({
            'drain': 'leftover',
            'stem_track_id': None,
            })

    def mark_merge(merged_track_id, stem_track_id):
        tracks[merged_track_id].update({
            'drain': 'merge',
            'stem_track_id': stem_track_id,
            })

    def mark_vanish(track_id):
        tracks[track_id].update({
            'drain': 'vanish',
            'stem_track_id': None,
            })


    for track_id, track in enumerate(tracks):
        if 'end_position' not in track:
            print(f"No end position: {track_id = }, {track = }")
        end_position = track['end_position']
        frame = track['end_frame']
        if end_position >= end_region:
            mark_disappeared_at_end(track_id)
            continue
        if end_position <= start_region:
            mark_disappeared_at_start(track_id)
            continue
        if frame == n_frames - 1:
            mark_leftover(track_id)
            continue
        nextt = peak_positions[frame + 1]
        if len(nextt):
            nearest_left = bisect_left(nextt, end_position) - 1
            nearest = (
                0 if nearest_left == -1 else
                nearest_left if nearest_left == len(nextt) - 1 else
                nearest_left if end_position - nextt[nearest_left] < nextt[nearest_left + 1] - end_position else
                nearest_left + 1
            )
            if -max_merge_distance <= end_position - peak_positions[frame + 1][nearest] <= max_merge_distance:
                mark_merge(track_id, track_ids[frame + 1][nearest])
                continue
        mark_vanish(track_id)




    tracks_df = pd.DataFrame(tracks)[
        [
            'track_id', 
            'tree_id',
            'start_frame',
            'end_frame',
            'start_position',
            'end_position',
            'source',
            'parent_track_id',
            'split_frame',
            'drain',
            'stem_track_id',
        ]
    ]#.fillna(pd.NA)

    for col in [
        'track_id', 
        'tree_id',
        'start_frame',
        'end_frame',
        'start_position',
        'end_position',
        'parent_track_id',
        'split_frame',
        'stem_track_id',
    ]:
        tracks_df[col] = tracks_df[col].astype(pd.Int32Dtype())
    tracks_df = tracks_df.set_index('track_id')




    return track_ids, tracks_df




