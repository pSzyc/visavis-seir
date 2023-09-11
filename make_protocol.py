import numpy as np

from jinja2 import Template



with open('./template.protocol') as f:
    template = Template(f.read())


pulse_intervals = np.random.randint(80, 120, size=250)
#pulse_intervals = np.arange(120, 80, -5)
pulse_times = list(np.cumsum(pulse_intervals))

simulation_periods = list(zip([0] + pulse_times, pulse_times))


with open('./out.protocol', 'w') as f:
    f.write(
        template.render(simulation_periods=simulation_periods)
    )
