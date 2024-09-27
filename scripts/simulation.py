# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
from pathlib import Path
from shutil import rmtree
import json

from typing import Optional

from .client import VisAVisClient
from .defaults import TEMP_DIR
from .make_protocol import make_protocol
from .utils import random_name

def run_simulation(
    
    parameters,
    channel_width,
    channel_length,
    pulse_intervals,
    logging_interval,
    seed=0,
    lattice_top_edge_aperiodic=False,
    verbose=False,
    states=False,
    activity=True,
    images=False,
    clean_up=True,
    save_states=False,
    save_activity=False,
    sim_root: Path | str = Path(TEMP_DIR) / 'qeirq',
    sim_dir_name: Optional[str] = None,
    outdir=None,
    ):

    sim_root = Path(sim_root)
    if sim_dir_name is None:
        sim_dir_name = random_name(12)

    (sim_root / sim_dir_name).mkdir(exist_ok=True, parents=True)
    
    sim_temp_results_dir = sim_root / sim_dir_name / 'simulation_results'
    if sim_temp_results_dir.exists():
        rmtree(str(sim_temp_results_dir))

        
    client = VisAVisClient(sim_root=sim_root)

    protocol_file_path = make_protocol(
        pulse_intervals=pulse_intervals,
        logging_interval=logging_interval,
        out_folder=sim_root / sim_dir_name,
        lattice_top_edge_aperiodic=lattice_top_edge_aperiodic,
    )

    result = client.run(
        parameters_json=parameters,
        channel_length=channel_length,
        channel_width=channel_width,
        protocol_file_path=protocol_file_path,
        verbose=verbose,
        dir_name=sim_dir_name + '/' + sim_temp_results_dir.name,
        seed=seed,
        states=states,
        activity=activity,
        images=images,
        clean_up=clean_up
    )

    if clean_up:
        rmtree(str(sim_root / sim_dir_name))


    if outdir and states and save_states:
        with open (outdir / 'input_protocol.json', 'w') as file:
            json.dump(pulse_intervals, file)
        result.states.to_csv(outdir / 'simulation_results.csv')     
    if outdir and save_activity:
        outdir.absolute().mkdir(parents=True, exist_ok=True)
        result.activity.to_csv(outdir / 'activity.csv')     


    return result
