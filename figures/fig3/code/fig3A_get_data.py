from pathlib import Path
from shutil import rmtree

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py

from scripts.client import VisAVisClient, _random_name
from scripts.make_protocol import make_protocol
from scripts.defaults import TEMP_DIR, PARAMETERS_DEFAULT


def generate_data(channel_width, channel_length, interval, n_pulses, interval_after=None, duration=5, seed=0):
    
    sim_dir = Path(f"{TEMP_DIR}/visavis_seir/periodic/" + _random_name(12))

    client = VisAVisClient()

    pulse_intervals = n_pulses * [interval]
    if interval_after is None:
        interval_after = int(2 * 3.6*channel_length+200)
    protocol_file_path = make_protocol(
            pulse_intervals=list(pulse_intervals) + [interval_after],
            duration=duration,
            out_folder=sim_dir,
        )

    result = client.run(
            parameters_json=PARAMETERS_DEFAULT,
            width=channel_width,
            length=channel_length,
            protocol_file_path=protocol_file_path,
            verbose=False,
            activity=True,
            states=False,
            seed=seed,
        )
    rmtree(sim_dir)
    return result
    

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3A'
data_dir.mkdir(exist_ok=True, parents=True)

channel_widths = [6]
channel_lengths = [300]
# intervals = [100, 150, 200]
intervals = [180, 120, 60]
duration = 5

for channel_width in channel_widths:
    for channel_length in channel_lengths:
        for interval in intervals:
            result = generate_data(
                channel_length=channel_length,
                channel_width=channel_width,
                interval=interval,
                n_pulses=4200//interval,
                interval_after=0,
                duration=duration,
                seed=2,
            )
            result.activity.to_csv(data_dir / f'fig3A_w-{channel_width}-l-{channel_length}-interval-{interval}--activity.csv')

