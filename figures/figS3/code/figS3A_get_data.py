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
from scripts.defaults import PARAMETERS_DEFAULT, MOL_STATES_DEFAULT, TEMP_DIR
from scripts.utils import simple_starmap, starmap, compile_if_not_exists



data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3' / 'figS3A' / 'approach1'
data_dir.mkdir(exist_ok=True, parents=True)



def run_single(
    client,
    sim_root,
    pulse_intervals,
    simulation_id,
    parameters=PARAMETERS_DEFAULT,
    mol_states=MOL_STATES_DEFAULT,
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
            mol_states_json=mol_states,
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

        tracks, input_pulse_to_tree_id = determine_fates(
            result.activity,
            input_protocol=pulse_intervals,
            duration=duration,
            channel_length=channel_length,
            outdir=outdir and outdir / f'sim-{simulation_id}',
            verbose=False,
            plot_results=plot_results,
            save_csv=save_iterations,
            indir=indir and indir / f'sim-{simulation_id}',
            returns=['tracks', 'input_pulse_to_tree_id'],
            **kwargs,
            )

        # first_pulse_position = tracks[tracks['tree_id'].eq(input_pulse_to_tree_id[0])].groupby('seconds')['h'].max().reindex(range(0,sum(pulse_intervals), duration))
        # second_pulse_position = tracks[tracks['tree_id'].eq(input_pulse_to_tree_id[1])].groupby('seconds')['h'].max().reindex(range(0,sum(pulse_intervals), duration))

        first_pulse_track = tracks[tracks['tree_id'].eq(input_pulse_to_tree_id[0])]
        second_pulse_track = tracks[tracks['tree_id'].eq(input_pulse_to_tree_id[1])]
        first_time_of_reaching = pd.Series([first_pulse_track[first_pulse_track['h'] >= h]['seconds'].min() for h in range(channel_length)], index=range(channel_length))
        second_time_of_reaching = pd.Series([second_pulse_track[second_pulse_track['h'] >= h]['seconds'].min() for h in range(channel_length)], index=range(channel_length))

        first_time_of_reaching.index.name = 'h'
        second_time_of_reaching.index.name = 'h'
        # common_track_frames = int(common_track_length / duration)


        return first_time_of_reaching, second_time_of_reaching




def generate_dataset(
    interval,
    n_simulations,
    parameters=PARAMETERS_DEFAULT,
    mol_states=MOL_STATES_DEFAULT,
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
        reaching_times = pd.read_csv(outdir / 'reaching_times.csv').set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'h'])
        # first_time_of_reaching = pd.read_csv(outdir / 'first_time_of_reaching.csv').set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'h'])['seconds']
        # second_time_of_reaching = pd.read_csv(outdir / 'second_time_of_reaching.csv').set_index(['channel_width', 'channel_length', 'interval', 'simulation_id', 'h'])['seconds']
        return reaching_times

    visavis_bin = compile_if_not_exists(channel_width, channel_length)
    if outdir:
        outdir.mkdir(exist_ok=True, parents=True)
    
    sim_root = Path(TEMP_DIR) / 'tracking' 
    client = VisAVisClient(
        visavis_bin=visavis_bin,
        sim_root=sim_root,
    )

    pulse_intervals = [interval, interval_after]

    first_time_of_reaching, second_time_of_reaching = list(zip(*starmap(
        run_single,
        [dict(
            client=client,
            sim_root=sim_root,
            pulse_intervals=pulse_intervals,
            simulation_id=simulation_id,
            parameters=parameters,
            mol_states=mol_states,
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

    assert not (interval % duration), f"{interval = } must be divisible by {duration = }"

    first_time_of_reaching = pd.DataFrame(first_time_of_reaching)
    first_time_of_reaching.index.name = 'simulation_id'
    first_time_of_reaching['channel_width'] = channel_width
    first_time_of_reaching['channel_length'] = channel_length
    first_time_of_reaching['interval'] = interval
    first_time_of_reaching = first_time_of_reaching.reset_index().set_index(['channel_width', 'channel_length', 'interval', 'simulation_id'])
    first_time_of_reaching = first_time_of_reaching.stack()
    first_time_of_reaching.name = 'first_pulse_reaching_time'


    second_time_of_reaching = pd.DataFrame(second_time_of_reaching)
    second_time_of_reaching.index.name = 'simulation_id'
    second_time_of_reaching['channel_width'] = channel_width
    second_time_of_reaching['channel_length'] = channel_length
    second_time_of_reaching['interval'] = interval
    second_time_of_reaching = second_time_of_reaching.reset_index().set_index(['channel_width', 'channel_length', 'interval', 'simulation_id'])
    second_time_of_reaching = second_time_of_reaching.stack()
    second_time_of_reaching.name = 'second_pulse_reaching_time'
  
    reaching_times = pd.DataFrame([first_time_of_reaching, second_time_of_reaching]).T

    if outdir:
        reaching_times.to_csv(outdir / 'reaching_times.csv')
        # first_time_of_reaching.to_csv(outdir / 'first_time_of_reaching.csv')
        # second_time_of_reaching.to_csv(outdir / 'second_time_of_reaching.csv')

    return reaching_times


def simulate(n_sim, channel_widths, channel_lengths, outdir, n_workers, per_width_kwargs={}, per_length_kwargs={}, **kwargs):

    reaching_times = pd.concat(simple_starmap(
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
        ))
    
    reaching_times.to_csv(outdir / 'reaching_times.csv')

    return reaching_times


channel_widths = [6]#[::-1]
intervals = [40,45,50,55,60,65,70,75,80,85,90,95,100,110,120,130,140,150,160,180,200]
channel_lengths = [1000]



reaching_times = simulate(
    n_sim=3000,
    channel_widths=channel_widths,
    channel_lengths=channel_lengths,
    outdir=data_dir,
    n_workers=20,
    use_cached=True,
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


(reaching_times['second_pulse_reaching_time'] - reaching_times['first_pulse_reaching_time']).groupby(['channel_width', 'channel_length', 'interval', 'h']).describe().to_csv(data_dir / 'difference_in_reaching_times.csv')
reaching_time_by_h = reaching_times.groupby(['channel_width', 'channel_length', 'interval', 'h']).mean()
reaching_time_by_h.to_csv(data_dir / 'reaching_time_by_h.csv')
# (second_time_of_reaching - first_time_of_reaching).groupby(['channel_width', 'channel_length', 'interval', 'h']).quantile(.1).to_csv(data_dir / 'difference_trajectories_q1.csv')
# (second_time_of_reaching - first_time_of_reaching).groupby(['channel_width', 'channel_length', 'interval', 'h']).min().to_csv(data_dir / 'difference_trajectories_min.csv')
# (second_time_of_reaching - first_time_of_reaching).groupby(['channel_width', 'channel_length', 'interval', 'h']).median().to_csv(data_dir / 'difference_trajectories_median.csv')



