from pathlib import Path
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
from shutil import rmtree

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.defaults import TEMP_DIR, PARAMETERS_DEFAULT, MOL_STATES_DEFAULT
from scripts.make_protocol import make_protocol
from scripts.tracking import determine_fates
from scripts.client import VisAVisClient, _random_name
from scripts.utils import simple_starmap, starmap, compile_if_not_exists

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1'  / 'approach5'
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

        track_info = determine_fates(
            result.activity,
            input_protocol=pulse_intervals,
            duration=duration,
            channel_length=channel_length,
            outdir=outdir and outdir / f'sim-{simulation_id}',
            verbose=False,
            plot_results=plot_results,
            save_csv=save_iterations,
            indir=indir and indir / f'sim-{simulation_id}',
            returns=['track_info'],
            **kwargs,
            )

        # data_part = pd.read_csv(outdir / f"sim-{simulation_id}" / 'pulse_fates.csv')
        selected_tracks = track_info[track_info['front_direction'].eq(1) & track_info['tracl_length'].gt(20)]
        displacement = (selected_tracks['track_end_position'] - selected_tracks['track_start_position']).sum()
        time = (selected_tracks['track_end'] - selected_tracks['track_start']).sum()

        return pd.DataFrame({'simulation_id': simulation_id, 'displacement': [displacement], 'time': [time]})



def generate_dataset(
    input_protocol,
    n_simulations,
    parameters=PARAMETERS_DEFAULT,
    mol_states=MOL_STATES_DEFAULT,
    channel_width=7,
    channel_length=300,
    duration=5,
    interval_after=1500,
    outdir=None,
    plot_results=False,
    save_states=True,
    save_iterations=True,
    indir=None,
    use_cached=False,
    n_workers=None,
    **kwargs
):

    if use_cached:
        return pd.read_csv(outdir / 'displacements.csv').set_index(['channel_width', 'channel_length', 'simulation_id'])
    visavis_bin = compile_if_not_exists(channel_width, channel_length)
    if outdir:
        outdir.mkdir(exist_ok=True, parents=True)
    
    sim_root = Path(TEMP_DIR) / 'tracking' 
    client = VisAVisClient(
        visavis_bin=visavis_bin,
        sim_root=sim_root,
    )

    pulse_intervals = list(input_protocol) + [interval_after]

    displacements = pd.concat(starmap(
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
            outdir=outdir,
            plot_results=plot_results,
            save_states=save_states,
            save_iterations=save_iterations,
            indir=indir,
            **kwargs,
        ) for simulation_id in range(n_simulations)],
        processes=n_workers,
    ))

    displacements['channel_width'] = channel_width
    displacements['channel_length'] = channel_length
    
    displacements = displacements.set_index(['channel_width','channel_length', 'simulation_id'])
    if outdir:
        displacements.to_csv(outdir / 'displacements.csv')
    return displacements

channel_lengths = [30]
channel_widths = (list(range(1,10,1)) + list(range(10,21,2)))#[::-1]
# channel_widths = [1]

displacements = pd.concat([generate_dataset(
    input_protocol=[],
    n_simulations=3000,
    channel_width=channel_width,
    channel_length=channel_length,
    duration=1,
    interval_after=1500,
    outdir=data_dir / f'l-{channel_length}-w-{channel_width}',
    plot_results=False,
    save_states=False,
    save_iterations=False,
    # use_cached=True,
    n_workers=10,
    exp_mean=0.01,
    min_peak_height=0.03 / channel_width,
    front_direction_minimal_distance=min(max(channel_width - 1, 1), 5),
) for channel_length, channel_width in product(channel_lengths, channel_widths)])

displacements_grouped = displacements.groupby(['channel_width', 'channel_length']).sum()
mean_velocity = displacements_grouped['displacement'] /  displacements_grouped['time']
mean_velocity.name = 'velocity'

displacements.to_csv(data_dir / 'displacements.csv')
displacements_grouped.to_csv(data_dir / 'displacements_grouped.csv')
mean_velocity.to_csv(data_dir / 'velocities.csv')

print(displacements)
print(displacements.groupby('channel_width').mean())
print(displacements.groupby('channel_width').sum().pipe(lambda df: df['time'] / df['displacement']))



