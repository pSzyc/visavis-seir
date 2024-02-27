from pathlib import Path
import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS3'/ 'figS3-3' / 'approach1'


c_field = 'channel_length'
x_field = 'channel_width'
normalize = True
df = pd.read_csv(data_dir / 'pulse_fates.csv')
df = df[df['fate'].eq('transmitted')]
for c_val, data in df.groupby(c_field):
    data = data.reset_index().set_index(['channel_length', 'channel_width'])

    var_data = (data['track_end'] / np.sqrt(data.index.get_level_values('channel_length') if normalize else 1)).groupby(x_field).var()
    plt.plot(var_data.index, var_data.values, 'o-', label=f'{c_val}')
plt.legend(title=c_field.replace('_', ' '))
# plt.yscale('log')
plt.savefig(Path(__file__).parent.parent / 'panels' / 'figS3-3.png')
plt.savefig(Path(__file__).parent.parent / 'panels' / 'figS3-3.svg')