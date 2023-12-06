from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from subplots_from_axsize import subplots_from_axsize
from sklearn.linear_model import LinearRegression
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.defaults import PARAMETERS_DEFAULT, MOL_STATES_DEFAULT


data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / 'approach2'
data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1'/ 'l-30'
data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2'/ 'fig2C' / 'approach7'
data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / 'approach3'
out_dir = Path(__file__).parent.parent / 'panels'
out_dir.mkdir(exist_ok=True, parents=True)

channel_lengths = [300]
channel_widths = (list(range(1,10,1)) + list(range(10,21,2)))
simulations = range(1000)

yaxis_step = .5
xmin = 10
xmax = 20
# bins_per_unit = 10
bins_per_unit = 2
duration = 5

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
plt.vlines([60 / (MOL_STATES_DEFAULT['n_e'] / PARAMETERS_DEFAULT['e_incr'] + .5 / PARAMETERS_DEFAULT['c_rate'])], 0.5, yaxis_step * (max(channel_widths) + 1.5), color='k', alpha=.4, ls=':')

plt.close()

fig, ax = subplots_from_axsize(1, 1, (45 / 25.4, 45 / 25.4))

pulse_fates_all = pd.concat([
        pd.read_csv(data_dir / f"w-{channel_width}-l-{channel_length}" / 'pulse_fates.csv') 
        for channel_length,channel_width in product(channel_lengths, channel_widths)
        ],
    ).set_index(['channel_length', 'channel_width'])
pulse_fates_all = pulse_fates_all[pulse_fates_all['track_end'] > 0]
front_velocities = (pulse_fates_all['track_end_position']-1) / (pulse_fates_all['track_end'] - .5 / PARAMETERS_DEFAULT['c_rate']) * 60
front_velocities.groupby('channel_width').mean().plot(marker='o', lw=1, ms=3)

analytical_velocity = 60 / (MOL_STATES_DEFAULT['n_e'] / PARAMETERS_DEFAULT['e_incr'] + .5 / PARAMETERS_DEFAULT['c_rate'])
ax.hlines([analytical_velocity], 0, 20, color='k', alpha=.4, ls=':')
ax.set_xlim(0, 21)
ax.set_ylim(0, 21)
ax.set_xlabel('channel width')
ax.set_ylabel('velocity [steps/h]')


plt.savefig(out_dir / 'figS1.svg')
plt.savefig(out_dir / 'figS1.png')


print(pulse_fates_all)
velocities = []
fig, axs = subplots_from_axsize(4,5,(2,2))
for ax, (channel_width, data) in zip(axs.flatten(), pulse_fates_all.groupby('channel_width')):
    counts = data.value_counts(['track_end_position', 'track_end']).reset_index()
    counts.plot.scatter('track_end_position', 'track_end', s=10*counts[0], alpha=0.1, ax=ax)
    
    lr = LinearRegression().fit(data[['track_end_position']].to_numpy(), data[['track_end']].to_numpy())
    xs = np.array([[0,300]]).T
    ys = lr.predict(xs)
    ax.plot(xs, ys)
    ax.set_title(f"y={lr.coef_} * x + {lr.intercept_}")
    print(f"w={channel_width}:  y={lr.coef_[0][0]} * x + {lr.intercept_[0]}")
    velocities.append(60/lr.coef_[0][0])

fig, ax = subplots_from_axsize(1,1,(3,3))
ax.plot(channel_widths, velocities, marker='o', lw=1, ms=3)
analytical_velocity = 60 / (MOL_STATES_DEFAULT['n_e'] / PARAMETERS_DEFAULT['e_incr'] + .5 / PARAMETERS_DEFAULT['c_rate'])
ax.hlines([analytical_velocity], 0, 20, color='k', alpha=.4, ls=':')
ax.set_xlim(0, 21)
ax.set_ylim(0, 21)
ax.set_xlabel('channel width')
ax.set_ylabel('velocity [steps/h]')

plt.show()