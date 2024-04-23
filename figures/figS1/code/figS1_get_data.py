import pandas as pd
import numpy as np
from pathlib import Path
# from matplotlib import pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.style import *
from subplots_from_axsize import subplots_from_axsize



channel_widths = list(range(1,10)) + list(range(10,21,2))

for channel_length, approach_S1, approach_2C in (
    (300, 'approach6', 'approach8'),
    (30, 'approach7', 'approach9'),
):

    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / approach_S1
    data_dir.mkdir(exist_ok=True, parents=True)
    fig2C_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / "fig2C" / approach_2C  



    data = pd.concat([
        pd.read_csv(fig2C_data_dir / f'w-{channel_width}-l-{channel_length}' / 'pulse_fates.csv', usecols=['channel_width', 'track_end', 'track_end_position', 'fate'])
        for channel_width in channel_widths
        ], ignore_index=True
    )
    data = data[(data['fate'] == 'transmitted') & (data['track_end_position'] >= channel_length-2)].set_index('channel_width')


    variance_per_step = data['track_end'].groupby('channel_width').var() / (channel_length - 1) # minus 1, as the front starts from h=1, not h=0
    variance_per_step.name = 'variance_per_step'
    variance_per_step.to_csv(data_dir  / 'variance_per_step.csv')

    time_per_step = data['track_end'].groupby('channel_width').mean() / (channel_length - 1)  # minus 1, as the front starts from h=1, not h=0
    time_per_step.name = 'time_per_step'
    time_per_step.to_csv(data_dir  / 'time_per_step.csv')

    velocity = 1 / (data['track_end'].groupby('channel_width').mean() / (channel_length - 1) )  # minus 1, as the front starts from h=1, not h=0
    velocity.name = 'velocity'
    velocity.to_csv(data_dir  / 'velocity.csv')

    # fig, axs = subplots_from_axsize(4,(len(channel_widths)-1)//4+1, (2,2), sharex=True)
    # for ax, (channel_width, data_part) in zip(axs.flatten(), data.groupby('channel_width')):
    #     data_part.plot.scatter('track_end', 'track_end_position', s=2, alpha=.1, ax=ax)
    #     # data_part.plot.hist('track_end_position', ax=ax, bins=30)

    #     ax.set_title(f'w = {channel_width}')

    # plt.show()
