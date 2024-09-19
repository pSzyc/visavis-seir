# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
from pathlib import Path
import pandas as pd

import json
from shutil import rmtree

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.defaults import PARAMETERS_DEFAULT, TEMP_DIR
from scripts.utils import simple_starmap, starmap, random_name
from scripts.tracked_results import TrackedResults
from scripts.simulation import run_simulation



def run_single(
    sim_root,
    pulse_intervals,
    simulation_id,
    channel_width,
    channel_length,
    parameters=PARAMETERS_DEFAULT,
    duration=5,
    outdir=None,
    save_states=False,
    save_activity=False,
    save_iterations=True,
    verbose=False,
    **kwargs):

        if outdir:
            sim_out_dir = outdir / f'sim-{simulation_id}'
            sim_out_dir.absolute().mkdir(parents=True, exist_ok=True)


        result = run_simulation(
            parameters=parameters,
            channel_width=channel_width,
            channel_length=channel_length,
            pulse_intervals=pulse_intervals,
            duration=duration,

            seed=19 + simulation_id,
            verbose=False,
            states=save_states,
            activity=True,
            save_states=save_states,
            save_activity=save_activity,
            sim_root= sim_root,
            sim_dir_name = f'w-{channel_width}--l-{channel_length}--sim-{simulation_id}--' + random_name(5),
            outdir=outdir and sim_out_dir,
        )

        tracked_results = TrackedResults(
            activity=result.activity,
            input_protocol=pulse_intervals,
            duration=duration,
            channel_length=channel_length,
            outdir=outdir and sim_out_dir,
            verbose=verbose,
            save_csv=save_iterations,
            **kwargs,
        )
        pulse_fates_part = tracked_results.pulse_fates
        pulse_fates_part['simulation_id'] = simulation_id
        return pulse_fates_part



def generate_dataset(
    input_protocol,
    n_simulations,
    channel_width=6,
    channel_length=300,
    interval_after=1500,
    plot_results=False,
    outdir=None,
    use_cached=False,
    n_workers=None,
    **kwargs
):

    if use_cached and outdir and (outdir / 'pulse_fates.csv').exists():
        pulse_fates = pd.read_csv(outdir / 'pulse_fates.csv').set_index(['channel_length', 'channel_width', 'simulation_id'])
        if len(pulse_fates.index.get_level_values('simulation_id').unique()) == n_simulations:
            return pulse_fates

    if outdir:
        outdir.mkdir(exist_ok=True, parents=True)
    
    sim_root = Path(TEMP_DIR) / 'qeir' / 'tracking' 

    pulse_intervals = list(input_protocol) + [interval_after]

    pulse_fates = pd.concat(starmap(
        run_single,
        [dict(
            sim_root=sim_root,
            pulse_intervals=pulse_intervals,
            simulation_id=simulation_id,
            channel_width=channel_width,
            channel_length=channel_length,
            plot_results=plot_results,
            use_cached=use_cached,
            outdir=outdir,
            **kwargs,
        ) for simulation_id in range(n_simulations)],
        processes=n_workers,
    ))

    pulse_fates['channel_width'] = channel_width
    pulse_fates['channel_length'] = channel_length
    
    if outdir:
        pulse_fates.set_index(['channel_length', 'channel_width', 'simulation_id']).to_csv(outdir / 'pulse_fates.csv')

    if plot_results and outdir:

        with open(outdir / 'kymographs.html', 'w') as kymo_html:
            kymo_html.write('<html><body>')
            kymo_html.write('''<style>
                            .overlay_title {
                                position: absolute;
                                top: 0;
                                left: 12;
                                color: yellow;
                            }
                            </style>''')
            for (_, _, simulation_id, pulse_id), pulse in pulse_fates.groupby(["channel_length", "channel_width", "simulation_id", "pulse_id"]):
                kymo_html.write(f'''
                                <div style="display: inline-block; position:relative"> 
                                <img src="sim-{simulation_id}/kymograph.png" alt="{simulation_id}:{pulse_id}" />
                                <span class="overlay_title">{simulation_id}:{pulse_id} {
                                    pulse.iloc[0][["fate", "reached_end", "reached_start"]].tolist()
                                    }</span></div>
                                    ''')
            kymo_html.write('</body></html>')
    return pulse_fates


def get_pulse_fate_counts(
        fate_criterion='reached',
        outdir=None,
        **kwargs,
        ):

    fields_to_groupby = (
        ['fate', 'forward', 'backward'] if fate_criterion == 'generated' else 
        ['fate', 'reached_end', 'reached_start'] if fate_criterion == 'reached' else
        []
    )
    counts = generate_dataset(
        outdir=outdir,
        **kwargs
        ).value_counts(['channel_length', 'channel_width'] + fields_to_groupby).sort_index()
    
    counts = pd.DataFrame({'count': counts})
    if outdir:
        counts.to_csv(outdir / 'pulse_fate_count.csv')

    return counts

