# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details 

from pathlib import Path
from shutil import rmtree

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py

from scripts.utils import random_name
from scripts.make_protocol import make_protocol
from scripts.simulation import run_simulation
from scripts.defaults import TEMP_DIR, PARAMETERS_DEFAULT


def generate_data(channel_width, channel_length, interval, n_pulses, parameters=PARAMETERS_DEFAULT,  interval_after=None, logging_interval=5, seed=0):
    
    sim_dir = Path(f"{TEMP_DIR}/qeir/periodic/" + random_name(12))

    if interval_after is None:
        interval_after = int(2 * 3.6*channel_length+200)

    result = run_simulation(
        parameters=parameters,
        channel_width=channel_width,
        channel_length=channel_length,
        pulse_intervals=n_pulses * [interval] + [interval_after],
        logging_interval=logging_interval,

        seed=seed,
        sim_root=sim_dir,
        )

    rmtree(sim_dir)

    return result
    

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig6'/ 'fig6BC' / 'approach1'
data_dir.mkdir(exist_ok=True, parents=True)

channel_width = 6
channel_length = 300
logging_interval = 5
simulation_length = 3000

data_sets = list(
    (150, {'r_forward_rate': 4/tau_r})
    for tau_r in [60,50,40,30,20]
) + list(
    (150, {'e_forward_rate': 4/tau_e})
    for tau_e in [4,6,8,10,12]
)


for interval, parameters_update in data_sets:

    result = generate_data(
        channel_length=channel_length,
        channel_width=channel_width,
        interval=interval,
        parameters={
            **PARAMETERS_DEFAULT,
            **parameters_update,
        },
        n_pulses=simulation_length //interval,
        interval_after=0,
        logging_interval=logging_interval,
        seed=2,
    )
    outdir = data_dir / '/'.join(map(lambda param_upd: f"{param_upd[0]}/{param_upd[1]:.3f}", sorted(parameters_update.items())))
    outdir.mkdir(exist_ok=True, parents=True)
    result.activity.to_csv(outdir / f'activity.csv')

