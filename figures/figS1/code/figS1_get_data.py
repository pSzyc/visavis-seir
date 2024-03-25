import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.style import *

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / "approach6"
fig2C_data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'fig2' / "fig2C" / 'approach8'    
data_dir.mkdir(exist_ok=True, parents=True)


channel_length = 300
channel_widths = list(range(1,10)) + list(range(10,21,2))

data = pd.concat([
    pd.read_csv(fig2C_data_dir / f'w-{channel_width}-l-{channel_length}' / 'pulse_fates.csv', usecols=['channel_width', 'track_end', 'fate'])
    for channel_width in channel_widths
    ], ignore_index=True
)
data = data[data['fate'] == 'transmitted']

variance_per_step = data.groupby('channel_width')['track_end'].var() / channel_length
variance_per_step.name = 'variance_per_step'
variance_per_step.to_csv(data_dir  / 'variance_per_step.csv')

time_per_step = data.groupby('channel_width')['track_end'].mean() / channel_length
time_per_step.name = 'time_per_step'
time_per_step.to_csv(data_dir  / 'time_per_step.csv')

velocity = 1 / (data.groupby('channel_width')['track_end'].mean() / channel_length)
velocity.name = 'velocity'
velocity.to_csv(data_dir  / 'velocity.csv')

