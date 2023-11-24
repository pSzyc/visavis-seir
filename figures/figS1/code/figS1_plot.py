from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.defaults import PARAMETERS_DEFAULT, STATES_DEFAULT


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / 'approach2'
data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1'/ 'l-30'
out_dir = Path(__file__).parent.parent / 'panels'
out_dir.mkdir(exist_ok=True, parents=True)

channel_lengths = [30]
channel_widths = (list(range(1,10,1)) + list(range(10,21,2)))
simulations = range(1000)

yaxis_step = .5
xmin = 10
xmax = 20
# bins_per_unit = 10
bins_per_unit = 2

for channel_length in channel_lengths:
    for it, channel_width in enumerate(channel_widths):
        width_dir = data_dir / f"w-{channel_width}-l-{channel_length}"

        pulse_fates_all = pd.read_csv(width_dir / 'pulse_fates.csv') 

        
        front_velocities = (pulse_fates_all['track_end_position']-1) / (pulse_fates_all['track_end'] - 2) * 60
        plt.figure(1)
        front_velocities.plot.hist(bins=np.linspace(xmin, xmax, (xmax-xmin) * bins_per_unit + 1), histtype='stepfilled', density=True, bottom=(channel_width-.5)*yaxis_step) #facecolor='none', edgecolor=f"C{it}",
        plt.annotate(f"{front_velocities.mean():.2f} $\pm$ {front_velocities.std():.2f}", (xmin+.02, (channel_width-.5)*yaxis_step))

        front_velocities_selected = front_velocities[
            # pulse_fates_all['track_end_position'] > 5
            pulse_fates_all['fate'] == 'transmitted'
            ]
        plt.figure(2)
        front_velocities_selected.plot.hist(bins=np.linspace(xmin, xmax, (xmax-xmin) * bins_per_unit + 1), histtype='stepfilled', density=True,  bottom=(channel_width-.5)*yaxis_step)
        if len(front_velocities_selected):
            plt.annotate(f"{front_velocities_selected.mean():.2f} $\pm$ {front_velocities_selected.std():.2f}", (xmin +.02, (channel_width-.5)*yaxis_step))
        plt.hlines([(channel_width-.5)*yaxis_step], xmin=xmin, xmax=xmax, color=f"C{it}", lw=1)



plt.figure(1)
plt.yticks([(channel_width-.5)*yaxis_step for channel_width in channel_widths], channel_widths)
# plt.hlines([(channel_width-.5)*yaxis_step for channel_width in channel_widths], xmin=0, xmax=0.35)
plt.ylabel('channel width')
plt.xlabel('front velocity [steps/h]')

plt.figure(2)
plt.yticks([(channel_width-.5)*yaxis_step for channel_width in channel_widths], channel_widths)
# plt.hlines([(channel_width-.5)*yaxis_step for channel_width in channel_widths], xmin=0, xmax=0.35)
plt.ylabel('channel width')
plt.xlabel('front velocity [steps/h]')
plt.vlines([60 / (STATES_DEFAULT['n_e'] / PARAMETERS_DEFAULT['e_incr'] + .5 / PARAMETERS_DEFAULT['c_rate'])], 0.5, yaxis_step * (max(channel_widths) + 1.5), color='k', alpha=.4, ls=':')



plt.savefig(out_dir / 'figS1.svg')
plt.savefig(out_dir / 'figS1.png')


plt.show()