# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
import numpy as np
import pandas as pd

def get_hierarchy(significant_splits, track_info, maximal_distance_in_event=20, maximal_interval_in_event=150):
    hierarchy = pd.DataFrame(         
        index=significant_splits.index if len(significant_splits.index) else [],
        columns=['parent_significant_split', 'first_child_significant_split', 'second_child_significant_split', 'is_same_event_as_parent', 'event_id'],
        # dtype=float,
        )
    hierarchy['is_same_event_as_parent'] = False
    for track_id, row in significant_splits.iterrows():
        up_track_id = track_id
        while True:
            prev_track_id = up_track_id
            up_track_id = track_info.loc[up_track_id, 'parent_track_id']
            if np.isnan(up_track_id):
                hierarchy.loc[track_id, 'parent_significant_split'] = np.nan
                break
            if up_track_id in significant_splits.index:
                hierarchy.loc[track_id, 'parent_significant_split'] = up_track_id
                hierarchy.loc[track_id, 'is_same_event_as_parent'] = (
                        (np.abs(significant_splits.loc[up_track_id, 'significant_split_position'] - significant_splits.loc[track_id, 'significant_split_position']) < maximal_distance_in_event)
                        & (np.abs(significant_splits.loc[up_track_id, 'significant_split_time'] - significant_splits.loc[track_id, 'significant_split_time']) < maximal_interval_in_event)
                    )
                if track_info.loc[up_track_id, 'first_child_track_id'] == prev_track_id:
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


def get_effective_front_directions(track_info):
    effective_front_direction = track_info['front_direction'] * track_info['is_significant'] * (track_info['track_length'] >= 40/5)
    effective_parent = pd.Series(track_info.index, index=track_info.index, name='effective_parent')

    child_track_info = track_info
    while len(child_track_info):
        the_child = child_track_info['first_child_track_id'].where(child_track_info['is_first_child_significant'] & ~child_track_info['is_second_child_significant'],
                    child_track_info['second_child_track_id'].where(~child_track_info['is_first_child_significant'] & child_track_info['is_second_child_significant'],
                    np.nan
                )).dropna()
        
        the_effective_parent = the_child.index
        effective_parent[the_child] = the_effective_parent

        child_track_info = track_info.loc[the_child]
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
         
        

def get_split_events(significant_splits_with_hierarchy, effective_front_directions, track_info):
    significant_splits_with_hierarchy = significant_splits_with_hierarchy.join(effective_front_directions)
    event_ids = significant_splits_with_hierarchy['event_id'].unique().astype(int)
    events = pd.DataFrame(index=event_ids, columns=['parent_track_id', 'parent_front_direction', 'forward', 'uncertain', 'backward', 'absolutely_forward', 'absolutely_backward'])
    events.index.name = 'event_id'
    for event_id, splits in significant_splits_with_hierarchy.groupby('event_id'):
        parent_track_id = splits[~splits['is_same_event_as_parent']].index[0]
        parent_front_direction = splits.loc[parent_track_id, 'effective_front_direction']

        front_direction_counts = (
            splits[splits['is_first_child_free']].join(effective_front_directions, on='first_child_track_id', lsuffix='_parent').value_counts('effective_front_direction').reindex([1,0,-1], fill_value=0)
            + splits[splits['is_second_child_free']].join(effective_front_directions, on='second_child_track_id', lsuffix='_parent').value_counts('effective_front_direction').reindex([1,0,-1], fill_value=0)
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