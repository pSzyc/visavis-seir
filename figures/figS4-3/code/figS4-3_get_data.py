from pathlib import Path
import pandas as pd
from itertools import product
from shutil import rmtree

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.client import VisAVisClient, _random_name
from scripts.make_protocol import make_protocol
from scripts.tracking import determine_fates
from scripts.defaults import PARAMETERS_DEFAULT, TEMP_DIR
from scripts.utils import simple_starmap, starmap



data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS4-3' / 'figS4-3' / 'approach1'
data_dir.mkdir(exist_ok=True, parents=True)



def run_single(
    client,
    sim_root,
    pulse_intervals,
    simulation_id,
    parameters=PARAMETERS_DEFAULT,
    channel_width=7,
    channel_length=300,
    duration=5,
    common_track_length=300,
    n_margin=0,
    outdir=None,
    plot_results=False,
    save_states=True,
    save_iterations=True,
    indir=None,
    **kwargs):

        sim_dir_name = f'w-{channel_width}--l-{channel_length}--sim-{simulation_id}' +  _random_name(5)
        (sim_root / sim_dir_name).mkdir(exist_ok=True, parents=True)

        protocol_file_path = make_protocol(
            pulse_intervals=pulse_intervals,
            duration=duration,
            out_folder=sim_root / sim_dir_name,
        )

        result = client.run(
            parameters_json=parameters,
            width=channel_width,
            length=channel_length,
            protocol_file_path=protocol_file_path,
            verbose=False,
            dir_name=sim_dir_name + '/' +  _random_name(5),
            seed=19 + simulation_id,
            states=save_states,
            activity=True,
        )
        rmtree(str(sim_root /sim_dir_name))

        if outdir and save_states:
            sim_out_dir = outdir / f'sim-{simulation_id}'
            sim_out_dir.absolute().mkdir(parents=True, exist_ok=True)
            with open (sim_out_dir / 'input_protocol.json', 'w') as file:
                json.dump(pulse_intervals, file)

            result.states.to_csv(sim_out_dir / 'simulation_results.csv')     

        significant_splits, tracks, track_info = determine_fates(
            result.activity,
            input_protocol=pulse_intervals,
            duration=duration,
            channel_length=channel_length,
            outdir=outdir and outdir / f'sim-{simulation_id}',
            verbose=False,
            plot_results=plot_results,
            save_csv=save_iterations,
            indir=indir and indir / f'sim-{simulation_id}',
            returns=['significant_splits', 'tracks', 'track_info'],
            **kwargs,
            )

        print(len(significant_splits), end=' ')
            

        ss_ids = []
        first_trajectories_parts = []
        second_trajectories_parts = []

        common_track_frames = int(common_track_length / duration)

        for ss_id, (parent_track_id, significant_split) in enumerate(significant_splits.iterrows()):
            first_branch_tracks = []
            track_id = int(significant_split['first_child_track_id'])
            while True:
                first_branch_tracks.append(track_id)
                the_track_info = track_info.loc[track_id]
                if the_track_info['is_first_child_significant'] == True:
                    track_id = int(the_track_info['first_child_track_id'])
                elif the_track_info['is_second_child_significant'] == True:
                    track_id = int(the_track_info['second_child_track_id'])
                else: 
                    break

            second_branck_tracks = []
            track_id = int(significant_split['second_child_track_id'])
            while True:
                second_branck_tracks.append(track_id)
                the_track_info = track_info.loc[track_id]
                if the_track_info['is_first_child_significant'] == True:
                    track_id = int(the_track_info['first_child_track_id'])
                elif the_track_info['is_second_child_significant'] == True:
                    track_id = int(the_track_info['second_child_track_id'])
                else:
                    break
            
            first_branch_trajectory = tracks[tracks['track_id'].isin(first_branch_tracks)]['h']
            second_branch_trajectory = tracks[tracks['track_id'].isin(second_branck_tracks)]['h']

            print(f':{len(first_branch_trajectory)},{len(second_branch_trajectory)}', end=' ')
            if len(first_branch_trajectory) < common_track_frames or len(second_branch_trajectory) < common_track_frames:
                continue

            ss_ids.append(ss_id)
            first_trajectories_parts.append(first_branch_trajectory.to_numpy()[:common_track_frames])
            second_trajectories_parts.append(second_branch_trajectory.to_numpy()[:common_track_frames])
            print(f'!', end=' ')


        first_trajectories = pd.DataFrame(first_trajectories_parts, columns=range(common_track_frames), index=ss_ids)
        second_trajectories = pd.DataFrame(second_trajectories_parts, columns=range(common_track_frames), index=ss_ids)

        first_trajectories.index.name = 'significant_split_id'
        second_trajectories.index.name = 'significant_split_id'

        first_trajectories['simulation_id'] = simulation_id
        second_trajectories['simulation_id'] = simulation_id

        return first_trajectories, second_trajectories




def generate_dataset(
    interval,
    n_simulations,
    parameters=PARAMETERS_DEFAULT,
    channel_width=7,
    channel_length=300,
    duration=5,
    interval_after=1500,
    n_margin=0,
    outdir=None,
    plot_results=False,
    save_states=False,
    save_iterations=True,
    indir=None,
    use_cached=False,
    n_workers=None,
    **kwargs
):

    if use_cached:
        first_trajectories = pd.read_csv(outdir / 'first_trajectories.csv').set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'significant_split_id'])
        second_trajectories = pd.read_csv(outdir / 'second_trajectories.csv').set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'significant_split_id'])
        return first_trajectories, second_trajectories

    if outdir:
        outdir.mkdir(exist_ok=True, parents=True)
    
    sim_root = Path(TEMP_DIR) / 'tracking' 
    client = VisAVisClient(sim_root=sim_root)

    pulse_intervals = [interval, interval_after]

    first_trajectories, second_trajectories = list(zip(*starmap(
        run_single,
        [dict(
            client=client,
            sim_root=sim_root,
            pulse_intervals=pulse_intervals,
            simulation_id=simulation_id,
            parameters=parameters,
            channel_width=channel_width,
            channel_length=channel_length,
            duration=duration,
            n_margin=n_margin,
            outdir=outdir,
            plot_results=plot_results,
            save_states=save_states,
            save_iterations=save_iterations,
            indir=indir,
            **kwargs,
        ) for simulation_id in range(n_simulations)],
        processes=n_workers,
    )))


    first_trajectories = pd.concat(first_trajectories)
    first_trajectories['channel_width'] = channel_width
    first_trajectories['channel_length'] = channel_length
    first_trajectories['interval'] = interval
    first_trajectories = first_trajectories.reset_index().set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'significant_split_id'])
    
    second_trajectories = pd.concat(second_trajectories)
    second_trajectories['channel_width'] = channel_width
    second_trajectories['channel_length'] = channel_length
    second_trajectories['interval'] = interval
    second_trajectories = second_trajectories.reset_index().set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'significant_split_id'])
    
    if outdir:
        first_trajectories.to_csv(outdir / 'first_trajectories.csv')
        second_trajectories.to_csv(outdir / 'second_trajectories.csv')

    return first_trajectories, second_trajectories


def simulate(n_sim, channel_widths, channel_lengths, outdir, n_workers, per_width_kwargs={}, per_length_kwargs={}, **kwargs):

    first_trajectories, second_trajectories = list(zip(*simple_starmap(
        generate_dataset,
        [
            dict(
                interval=interval,
                channel_length=l,
                n_simulations=n_sim,
                channel_width=w,
                outdir=outdir / f'w-{w}-l-{l}'/ f'interval-{interval}',
                n_workers=n_workers,
                ) | kwargs | (per_width_kwargs[w] if w in per_width_kwargs else {}) | (per_length_kwargs[l] if l in per_length_kwargs else {})
            for w in channel_widths for l in channel_lengths for interval in intervals
        ],
        )))

    first_trajectories = pd.concat(first_trajectories)
    second_trajectories = pd.concat(second_trajectories)

    return first_trajectories, second_trajectories


channel_widths = [6]#[::-1]
intervals = [40,60,90,120,200]
channel_lengths = [300]



first_trajectories, second_trajectories = simulate(
    n_sim=10,
    channel_widths=channel_widths,
    channel_lengths=channel_lengths,
    outdir=data_dir,
    n_workers=20,
    # use_cached=True,
    per_width_kwargs = {
        w: {
            'front_direction_minimal_distance': min(max(w - 1, 1), 5),
            'min_peak_height': 0.03 / w,
        } for w in channel_widths
    },
    per_length_kwargs={
        l: {
        'interval_after': int(l * 4.2),
        }
        for l in channel_lengths
    }
)

(second_trajectories - first_trajectories).groupby(['channel_width', 'channel_length', 'interval']).mean().to_csv(data_dir / 'difference_trajectories.csv')




