# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
 

from pathlib import Path
from shutil import rmtree

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py

from scripts.utils import random_name
from scripts.make_protocol import make_protocol
from scripts.simulation import run_simulation
from scripts.defaults import TEMP_DIR, PARAMETERS_DEFAULT


def generate_data(channel_width, channel_length, interval, n_pulses, interval_after=None, logging_interval=5, seed=0):
    
    sim_dir = Path(f"{TEMP_DIR}/qeir/periodic/" + random_name(12))

    if interval_after is None:
        interval_after = int(2 * 3.6*channel_length+200)

    result = run_simulation(
        parameters=PARAMETERS_DEFAULT,
        channel_width=channel_width,
        channel_length=channel_length,
        pulse_intervals=n_pulses * [interval] + [interval_after],
        logging_interval=logging_interval,

        seed=seed,
        sim_root=sim_dir,
        )

    rmtree(sim_dir)

    return result
    

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig3' / 'fig3A'
data_dir.mkdir(exist_ok=True, parents=True)

channel_widths = [6]
channel_lengths = [300]
# intervals = [100, 150, 200]
intervals = [180, 120, 60]
logging_interval = 5

for channel_width in channel_widths:
    for channel_length in channel_lengths:
        for interval in intervals:
            result = generate_data(
                channel_length=channel_length,
                channel_width=channel_width,
                interval=interval,
                n_pulses=4200//interval,
                interval_after=0,
                logging_interval=logging_interval,
                seed=2,
            )
            result.activity.to_csv(data_dir / f'fig3A_w-{channel_width}-l-{channel_length}-interval-{interval}--activity.csv')

