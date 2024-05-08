from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent)) # in order to be able to import from scripts.py

# from scripts.client import VisAVisClient
# from scripts.make_protocol import make_protocol
from scripts.tracking import determine_fates, get_pulse_positions
from scripts.plot_result import plot_result, plot_result_from_activity
#from scripts.tracker import make_tracks
from scripts.simulation import run_simulation

channel_width = 6#6
channel_length = 150#0

outpath = Path.cwd() / 'private' / 'current2'
outpath.mkdir(exist_ok=True, parents=True)

parameters = {
  "e_subcompartments_count": 4,
  "i_subcompartments_count": 2,
  "r_subcompartments_count": 4,
  "c_rate": 1/2,
  "e_forward_rate": 1,
  "i_forward_rate": 1,
  "r_forward_rate": 0.0667,
}

input_protocol = [500]*50
duration = 5

result = run_simulation(
    parameters=parameters,
    channel_width=channel_width,
    channel_length=channel_length,
    pulse_intervals=input_protocol,
    duration=duration,
    seed=0,
    verbose=False,
    states=False,
    activity=True,
    images=True,
    clean_up=False,
    save_states=False,
    save_activity=False,
    sim_root = Path.cwd() / 'private' / 'manual',
    sim_dir_name = None,
    outdir=None,
  )

# ffmpeg -framerate 24 -pattern_type glob -i '../private/manual/AAHOXEGCFWWE/simulation_results/lattice_00*.png' -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" out.mp4


# # result.states.to_csv(outpath / 'states.csv')

# pulse_positions_df = get_pulse_positions(result.activity, min_peak_height=0.03/width)
# pulse_positions = [
#   pulse_positions_df[pulse_positions_df['seconds'] == duration * frame]['h'].tolist()
#   for frame in range(len(result.activity))
# ]

# # print(pulse_positions)


# track_ids, tracks = make_tracks(pulse_positions, channel_length=length, max_split_distance=7+width//3, max_merge_distance=4+width//3)
# # print(track_ids)
# print(tracks)

# frame_s = [i for i, frame_tracks in enumerate(track_ids) for track_id in frame_tracks]
# track_id_s = [track_id for i, frame_tracks in enumerate(track_ids) for track_id in frame_tracks]
# pulse_position_s = [pulse_position for i, frame_positions in enumerate(pulse_positions) for pulse_position in frame_positions]

# pulse_positions_df['track_id'] = track_id_s
# pulse_positions_df['frame'] = frame_s
# pulse_positions_df.to_csv(outpath / 'pulse_positions.csv')

# track_positions = pulse_positions_df.set_index(['track_id', 'frame']).sort_index()


# plt.figure(figsize=(sum(input_protocol)/200,6))
# plt.scatter(frame_s, pulse_position_s, c=track_id_s, s=3, cmap='prism')
# tracks[tracks['drain'] == 'vanish'].plot.scatter('end_frame',   'end_position',   marker='x', s=20, c='olive', ax=plt.gca())
# tracks[tracks['drain'] == 'end']   .plot.scatter('end_frame',   'end_position',   marker='x', s=20, c='blue',  ax=plt.gca())
# tracks[tracks['drain'] == 'start'] .plot.scatter('end_frame',   'end_position',   marker='x', s=20, c='blue', ax=plt.gca())
# tracks[tracks['source'] == 'start'] .plot.scatter('start_frame',   'start_position', marker='^', s=20, c='blue', ax=plt.gca())
# tracks[tracks['source'] == 'end'] .plot.scatter('start_frame',   'start_position', marker='v', s=20, c='blue', ax=plt.gca())
# tracks[tracks['source'] == 'emerge'] .plot.scatter('start_frame',   'start_position', marker='+', s=20, c='pink', ax=plt.gca())
# # tracks[tracks['drain'] == 'merge'] .plot.scatter('end_frame',   'end_position',   marker='o', s=20, edgecolors='olive', c='none', ax=plt.gca())
# # tracks[tracks['source'] == 'split'].plot.scatter('start_frame', 'start_position', marker='o', s=20, edgecolors='cyan',  c='none', ax=plt.gca())

# for _, row in tracks[tracks['drain'] == 'merge'].iterrows():
#   plt.plot(
#     [row['end_frame'], row['end_frame'] + 1],
#     [row['end_position'], track_positions['h'].loc[row['stem_track_id'], row['end_frame']+1]], color='red', lw=1)

# for _, row in tracks[tracks['source'] == 'split'].iterrows():
#   plt.plot(
#     [row['start_frame'], row['split_frame']],
#     [row['start_position'], track_positions['h'].loc[row['parent_track_id'], row['split_frame']]], color='cyan', lw=1)

# plt.savefig(outpath / 'out.svg')
# tracks.to_csv(outpath / 'track_info.csv')

# plot_result_from_activity(result.activity, outfile=outpath / 'kymograph.png')



# # print(result.states)
# # # activity = get_activity(result.states)
# # print(activity)
# # plt.imshow(activity.unstack(level='seconds').to_numpy(), cmap='binary', origin='lower')

# # plt.savefig('kymograph.png')
# # plt.savefig('kymograph.svg')
# # maxima = get_maxima(activity)
# # print(maxima)
# # maxima.to_csv('maxima.csv')
# # maxima.to_pickle('maxima.pkl')



# # pulse_fates = determine_fates(result.states, input_protocol, outdir=outpath)

# # print(pulse_fates.value_counts(['fate', 'forward', 'backward']).sort_index())

# # plot_result(result, outfile=outpath.absolute() / 'kymograph.png')

