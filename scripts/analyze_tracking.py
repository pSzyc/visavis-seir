from pathlib import Path

import pandas as pd

from tqdm import tqdm
import subprocess
import json
import os
from multiprocessing import Pool
from functools import partial

import sys
sys.path.insert(0, '..') # in order to be able to import from scripts.py
sys.path.insert(0, '.') # in order to be able to import from scripts.py

from scripts.client import VisAVisClient
from scripts.make_protocol import make_protocol
from scripts.tracking import determine_fates
from scripts.defaults import PARAMETERS_DEFAULT



def generate_dataset(
    input_protocol,
    n_simulations,
    parameters=PARAMETERS_DEFAULT,
    channel_width=7,
    channel_length=300,
    duration=5,
    interval_after=1500,
    offset=None,
    n_margin=2,
    outdir=None,
):
    
    path_to_compiling_script = './scripts/compile_visavis.sh'

    visavis_bin = f'./target/bins/vis-a-vis-{channel_width}-{channel_length}'
    if not Path(visavis_bin).exists() and Path(path_to_compiling_script).exists():
        subprocess.call(
            [
                path_to_compiling_script,
                '-l',
                str(channel_length),
                '-w',
                str(channel_width),
            ],
            stdout=subprocess.DEVNULL,
        )
    client = VisAVisClient(
        visavis_bin=visavis_bin,
        sim_root=Path('/run/user') / os.environ['USER'],
    )

    pulse_intervals = list(input_protocol) + [interval_after]

    if offset is None:
        offset = channel_length * 3.6
    
    data_parts = []
    for simulation_id in tqdm(range(n_simulations)):
        (outdir / f'sim-{simulation_id}').absolute().mkdir(parents=True, exist_ok=True)
        with open (outdir / f'sim-{simulation_id}' / 'input_protocol.json', 'w') as file:
            json.dump(pulse_intervals, file)
        protocol_file_path = make_protocol(
            pulse_intervals=pulse_intervals,
            duration=duration,
            out_folder='/tmp/',
        )

        result = client.run(
            parameters_json=parameters,
            protocol_file_path=protocol_file_path,
            verbose=False,
        )
        
        data_part = determine_fates(
            result.states,
            input_protocol=pulse_intervals,
            outdir=outdir and outdir / f'sim-{simulation_id}',
            verbose=False)
        if n_margin > 0:
            data_part = data_part.iloc[n_margin:-n_margin]

        data_part['simulation_id'] = simulation_id
        data_parts.append(data_part)

    data = pd.concat(data_parts)

    data['channel_width'] = channel_width
    data['channel_length'] = channel_length

    return data


def get_occurrences(
        input_protocol,
        n_simulations,
        parameters=PARAMETERS_DEFAULT,
        channel_width=7,
        channel_length=300,
        duration=5,
        interval_after=1500,
        offset=None,
        n_margin=2,
        outdir=None,
        ):
    counts = generate_dataset(
        input_protocol=input_protocol,
        n_simulations=n_simulations,
        parameters=parameters,
        channel_width=channel_width,
        channel_length=channel_length,
        duration=duration,
        interval_after=interval_after,
        offset=offset,
        n_margin=n_margin,
        outdir=outdir,
        ).value_counts(['fate', 'forward', 'backward']).sort_index().rename(columns={0: 'count'})
    
    if outdir:
        counts.to_csv(outdir / 'pulse_fate_count.csv')
    return counts

def with_expanded_kwargs(fn, kwargs):
    return fn(**kwargs)


if __name__ == '__main__':

    outdir = Path('../private/fates')

    input_protocol = []
    
    channel_lengths = [300]
    channel_widths = list(range(1,10)) + list(range(10,21,2))
    
    with Pool(12) as pool:
        data_parts = pool.starmap(with_expanded_kwargs, ((get_occurrences, dict(
                input_protocol=input_protocol,
                n_simulations=1000,
                channel_length=channel_length,
                channel_width=channel_width,
                outdir=outdir / f"w-{channel_width}-l-{channel_length}",
                n_margin=0,
        )) for channel_length in channel_lengths  for channel_width in channel_widths[8:]
        ), chunksize=1)

    # data_parts = []
    # for channel_length in channel_lengths:
    #     for channel_width in channel_widths:
    #         data_part = get_occurrences(
    #             input_protocol=input_protocol,
    #             n_simulations=1000,
    #             channel_length=channel_length,
    #             channel_width=channel_width,
    #             outdir=outdir / f"w-{channel_width}-l-{channel_length}",
    #             n_margin=0,
    #             )
    #         data_parts.append(data_part)

    data = pd.concat(data_parts)
    data.to_csv(outdir / 'pulse_fate_count.csv')
    


