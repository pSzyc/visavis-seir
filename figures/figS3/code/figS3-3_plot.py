from pathlib import Path
import sys
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3'/ 'figS3-3' / 'approach1'

df = pd.read_csv(data_dir / 'pulse_fates.csv', usecols=['track_end', 'fate', 'channel_length', 'channel_width'])
df = df[df['fate'] == 'transmitted']
for channel_width in df['channel_width'].unique():
    var_data = df[df['channel_width'] == channel_width].groupby('channel_length')['track_end'].var()
    plt.plot(var_data.index, var_data.values, 'o-', label=f'w={channel_width}')
plt.legend()
plt.savefig(Path(__file__).parent.parent / 'panels' / 'figS3-3.png')