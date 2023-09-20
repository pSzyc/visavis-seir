from laptrack import LapTrack
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

def get_activations_at_h(img, h, plot = False, roll = 10):
    roll_start = img.iloc[h].rolling(roll, center=True, win_type='gaussian', min_periods = 1).mean(std=roll/6) # img indexed by position in channel 
    roll_start_diff = roll_start.diff()
    activations_start = roll_start[(roll_start_diff.fillna(np.inf) >= 0) & (roll_start_diff.shift(-1).fillna(-np.inf) < 0) & (roll_start  > 0)]    
    if plot:
        plt.scatter(activations_start.index, activations_start, color = 'red')
        plt.plot(roll_start)
        plt.show()
    return activations_start

def get_activations_time(img, plot = False, roll = 10, std_factor = 1/6):
    roll_start = img.rolling(roll, center=True, win_type='gaussian', min_periods = 1).mean(std=std_factor* roll) # img indexed by position in channel 
    roll_start_diff = roll_start.diff()
    activations_start = roll_start[(roll_start_diff.fillna(np.inf) >= 0) & (roll_start_diff.shift(-1).fillna(-np.inf) < 0) & (roll_start  > 0)]    
    df_activations = activations_start.reset_index().melt(
        id_vars=["h"],
        var_name="seconds", 
        value_name="peak",
    ).dropna()
    df_activations['seconds'] = df_activations['seconds'].astype(int)
    if plot:
        plt.scatter(df_activations.h, df_activations.seconds, color = 'black', s=0.75)
        plt.show()
    return df_activations

def get_infected_img(df):
    img_E = df.groupby(['seconds', 'h'])['E'].sum().unstack().T
    img_I = df.groupby(['seconds', 'h'])['I'].sum().unstack().T
    img = img_E + img_I
    return img

def get_tracks(img, lt, eps = 1e-3, roll = 20, std_factor = 1/6):
    df_activations = get_activations_time(img, 0, roll=roll, std_factor=std_factor)
    df_activations['seconds'] = df_activations['seconds'].astype(int)
    df_activations['seconds']//=4
    df_activations['h'] += eps * df_activations['seconds']
    track_df, split_df, merge_df = lt.predict_dataframe(
        df_activations[['h', 'seconds']].sort_values(by='seconds'),
        coordinate_cols=["h"],  
        frame_col="seconds",
        only_coordinate_cols=False,  
    )
    return (track_df, split_df, merge_df)

def plot_tracks(track_df, split_df, merge_df, color_by = 'tree_id'):
    frames = track_df.index.get_level_values("frame")
    frame_range = [frames.min(), frames.max()]
    k1, k2 = "seconds", "h"
    keys = [k1, k2]

    def get_track_end(track_id, first=True):
        df = track_df[track_df["track_id"] == track_id].sort_index(level="frame")
        return df.iloc[0 if first else -1][keys]

    for track_id, grp in track_df.groupby("track_id"):
        df = grp.reset_index().sort_values("frame")
        n = len(track_df['tree_id'].unique())
        colors = [list(np.random.uniform(0,1,3)) for _ in range(n)]
        cmap = matplotlib.colors.ListedColormap(colors)
        plt.scatter(df[k1], df[k2], c=df[color_by], vmin=0, vmax=n, cmap=cmap)#, vmin=frame_range[0], vmax=frame_range[1])
        for i in range(len(df) - 1):
            pos1 = df.iloc[i][keys]
            pos2 = df.iloc[i + 1][keys]
            plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], "-k")
        for _, row in list(split_df.iterrows()) + list(merge_df.iterrows()):
            pos1 = get_track_end(row["parent_track_id"], first=False)
            pos2 = get_track_end(row["child_track_id"], first=True)
            plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], "-k")
    
def get_events_df(track_df, split_df):
    df = pd.pivot_table(track_df, 'h', ['track_id', 'seconds'])
    front_speed = df['h'].groupby('track_id').diff().groupby('track_id').mean()
    front_speed.name = 'front_speed'
    front_direction = (front_speed > 0) * 2 - 1
    front_direction.name = 'front_direction'
    track_start = track_df.groupby('track_id')['seconds'].min()
    tree_df = track_df.value_counts(['track_id', 'tree_id']).index.to_frame(index=False).set_index('track_id')   
    events_df = split_df.join(front_direction, on='parent_track_id')\
        .join(front_direction, on='child_track_id', lsuffix='_parent', rsuffix='_child')\
            .join(track_start, on='child_track_id')\
                .join(tree_df, on='child_track_id')  
    return events_df