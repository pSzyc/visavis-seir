# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk

    
from typing import List
from jinja2 import Template
from pathlib import Path
import numpy as np

PROTOCOL_TEMPLATE = Template(
'''{% if lattice_top_edge_aperiodic %}lattice top edge aperiodic
{% endif %}{% for t0, t1 in simulation_periods %}+front at column 0
run  {{t0}}s...{{t1}}s  [{{logging_interval}}s]
{% endfor %}
'''
)

def make_protocol(
    pulse_intervals: List[int],
    logging_interval: int,
    out_folder: str,
    lattice_top_edge_aperiodic: bool,
) -> str:
    pulse_times = list(np.cumsum(pulse_intervals))
    simulation_periods = list(zip([0] + pulse_times, pulse_times))

    Path(out_folder).mkdir(parents=True, exist_ok=True)
    protocol_file_path = f'{out_folder}/out.protocol'
    with open(protocol_file_path, 'w') as f:
        f.write(
            PROTOCOL_TEMPLATE.render(
                simulation_periods=simulation_periods,
                logging_interval=logging_interval,
                lattice_top_edge_aperiodic=lattice_top_edge_aperiodic,
            )
        )
    return protocol_file_path

# make_protocol([100, 100], 2, '.')
