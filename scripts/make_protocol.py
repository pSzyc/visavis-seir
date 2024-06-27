# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
import numpy as np
from jinja2 import Template
from pathlib import Path

TEMPLATE = Template(
'''{% for t0, t1 in simulation_periods %}+front at column 0
run  {{t0}}s...{{t1}}s  [{{duration}}s]
{% endfor %}
'''
)

def make_protocol(pulse_intervals, duration, out_folder):
    pulse_times = list(np.cumsum(pulse_intervals))
    simulation_periods = list(zip([0] + pulse_times, pulse_times))

    Path(out_folder).mkdir(parents=True, exist_ok=True)
    with open(f'{out_folder}/out.protocol', 'w') as f:
        f.write(
            TEMPLATE.render(simulation_periods=simulation_periods, duration=duration)
        )
    return f'{out_folder}/out.protocol'

# make_protocol([100, 100], 2, '.')