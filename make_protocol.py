import numpy as np
from jinja2 import Template
from pathlib import Path

def make_protocol(pulse_intervals, duration, out_folder, template = './template.protocol'):
    with open(template) as f:
        template = Template(f.read())

    pulse_times = list(np.cumsum(pulse_intervals))
    simulation_periods = list(zip([0] + pulse_times, pulse_times))

    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    with open(out_folder / 'out.protocol', 'w') as f:
        f.write(
            template.render(simulation_periods=simulation_periods, duration=duration)
        )