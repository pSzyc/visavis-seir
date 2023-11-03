from pathlib import Path

import pandas as pd

from tqdm import tqdm
import json
from shutil import rmtree

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.client import VisAVisClient, _random_name
from scripts.make_protocol import make_protocol
from scripts.tracking import determine_fates
from scripts.defaults import PARAMETERS_DEFAULT, TEMP_DIR
from scripts.utils import starmap, compile_if_not_exists



def generate_dataset(
    input_protocol,
    n_simulations,
    parameters=PARAMETERS_DEFAULT,
    channel_width=7,
    channel_length=300,
    duration=5,
    interval_after=1500,
    offset=None,
    n_margin=0,
    outdir=None,
    plot_results=False,
    save_states=True,
    save_iterations=True,
    **kwargs
):
    print(interval_after)
    visavis_bin = compile_if_not_exists(channel_width, channel_length)
    if outdir:
        outdir.mkdir(exist_ok=True, parents=True)
    
    sim_root = Path(TEMP_DIR) / 'tracking' 
    client = VisAVisClient(
        visavis_bin=visavis_bin,
        sim_root=sim_root,
    )

    pulse_intervals = list(input_protocol) + [interval_after]

    if offset is None:
        offset = channel_length * 3.6
    
    data_parts = []
    for simulation_id in tqdm(range(n_simulations)):

        sim_dir_name = f'w-{channel_width}--l-{channel_length}--sim-{simulation_id}'
        (sim_root / sim_dir_name).mkdir(exist_ok=True, parents=True)

        protocol_file_path = make_protocol(
            pulse_intervals=pulse_intervals,
            duration=duration,
            out_folder=sim_root / sim_dir_name,
        )

        result = client.run(
            parameters_json=parameters,
            protocol_file_path=protocol_file_path,
            verbose=False,
            dir_name=sim_dir_name + '/' +  _random_name(5),
            seed=19 + simulation_id,
        )
        rmtree(str(sim_root /sim_dir_name))

        if outdir and save_states:
            sim_out_dir = outdir / f'sim-{simulation_id}'
            sim_out_dir.absolute().mkdir(parents=True, exist_ok=True)
            with open (sim_out_dir / 'input_protocol.json', 'w') as file:
                json.dump(pulse_intervals, file)

            result.states.to_csv(sim_out_dir / 'simulation_results.csv')     

        data_part = determine_fates(
            result.states,
            input_protocol=pulse_intervals,
            outdir=outdir and outdir / f'sim-{simulation_id}',
            verbose=False,
            plot_results=plot_results,
            save_csv=save_iterations,
            **kwargs,
            )
        if n_margin > 0:
            data_part = data_part.iloc[n_margin:-n_margin]

        data_part['simulation_id'] = simulation_id
        data_parts.append(data_part)



    pulse_fates = pd.concat(data_parts)

    pulse_fates['channel_width'] = channel_width
    pulse_fates['channel_length'] = channel_length
    
    if plot_results and outdir:
        pulse_fates.set_index(['channel_length', 'channel_width', 'simulation_id']).to_csv(outdir / 'pulse_fates.csv')

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
                                <img src="sim-{simulation_id}/out-kymo.png" alt="{simulation_id}:{pulse_id}" />
                                <span class="overlay_title">{simulation_id}:{pulse_id} {
                                    pulse.iloc[0][["fate", "reached_end", "reached_start"]].tolist()
                                    }</span></div>
                                    ''')
            kymo_html.write('</body></html>')
    return pulse_fates


def get_pulse_fate_counts(
        input_protocol,
        n_simulations,
        parameters=PARAMETERS_DEFAULT,
        channel_width=7,
        channel_length=300,
        duration=5,
        interval_after=1500,
        fate_criterion='reached',
        offset=None,
        n_margin=0,
        outdir=None,
        plot_results=False,
        save_states=True,
        save_iterations=True,
        **kwargs,
        ):

    fields_to_groupby = (
        ['fate', 'forward', 'backward'] if fate_criterion == 'generated' else 
        ['fate', 'reached_end', 'reached_start'] if fate_criterion == 'reached' else
        []
    )
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
        plot_results=plot_results,
        save_states=save_states,
        save_iterations=save_iterations,
        **kwargs
        ).value_counts(['channel_length', 'channel_width'] + fields_to_groupby).sort_index()
    
    counts = pd.DataFrame({'count': counts})
    # counts['channel_width'] = channel_width
    # counts['channel_length'] = channel_length
    # counts = counts.set_index( + fields_to_groupby)
    if outdir:
        counts.to_csv(outdir / 'pulse_fate_count.csv')

    return counts


if __name__ == '__main__':

    outdir = Path('../private/fates/approach7')
    outdir.mkdir(exist_ok=True, parents=True)

    input_protocol = []
    
    channel_lengths = [300]
    # channel_widths = list(range(1,10)) + list(range(10,21,2))
    channel_widths = [10]
    
    data_parts = starmap(get_pulse_fate_counts, [
            dict(
                input_protocol=input_protocol,
                n_simulations=20,
                channel_length=channel_length,
                channel_width=channel_width,
                outdir=outdir / f"w-{channel_width}-l-{channel_length}",
                n_margin=0,
                interval_after=int(2.2 * channel_length * 3.6),
                plot_results=False,
                save_states=False,
                save_iterations=False,
            ) 
        for channel_length in channel_lengths for channel_width in channel_widths
        ])

    data = pd.concat(data_parts)
    data.to_csv(outdir / 'pulse_fate_count.csv')

