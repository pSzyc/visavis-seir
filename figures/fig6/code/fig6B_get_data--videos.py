from pathlib import Path
from shutil import rmtree

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py

from scripts.utils import random_name
from scripts.make_video import make_video
from scripts.defaults import PARAMETERS_DEFAULT


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS5' / 'approach1'
data_dir.mkdir(exist_ok=True, parents=True)

channel_width = 6
channel_length = 100
duration = 5


simulation_length = 2100
interval_after = 0

data_sets = list(
    (150, {'r_forward_rate': 4/tau_r})
    for tau_r in [60,40,30,20]
) + list(
    (150, {'e_forward_rate': 4/tau_e})
    for tau_e in [4,6,10]
)




for interval, parameters_update in data_sets[:1]:

    n_pulses = simulation_length // interval

    outdir = data_dir / '-'.join(map(lambda param_upd: f"{param_upd[0]}-{param_upd[1]:.3f}", sorted(parameters_update.items())))
    outdir.mkdir(exist_ok=True, parents=True)

    make_video(
        channel_length=channel_length,
        channel_width=channel_width,
        input_protocol=[interval] * n_pulses + [interval_after],
        parameters={
            **PARAMETERS_DEFAULT,
            **parameters_update,
        },
        duration=duration,
        seed=2,
        outdir=outdir,
    )

