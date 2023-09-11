import numpy as np
from jinja2 import Template
from pathlib import Path

def make_protocol(pulse_intervals, duration, out_folder):
    template = '''{% for t0, t1 in simulation_periods %}+batsoup 1
run  {{t0}}s...{{t1}}s  [{{duration}}s]
{% endfor %}
'''
    template = Template(template)

    pulse_times = list(np.cumsum(pulse_intervals))
    simulation_periods = list(zip([0] + pulse_times, pulse_times))

    Path(out_folder).mkdir(parents=True, exist_ok=True)
    with open(f'{out_folder}/out.protocol', 'w') as f:
        f.write(
            template.render(simulation_periods=simulation_periods, duration=duration)
        )
    return f'{out_folder}/out.protocol'