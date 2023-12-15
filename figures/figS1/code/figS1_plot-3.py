import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent)) # in order to be able to import from scripts.py
from scripts.style import *

data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'figS1' / "fig-3"
data = pd.read_csv(data_dir / 'pulse_fates.csv', usecols=['channel_width', 'track_end', 'fate'])
data = data[data['fate'] == 'transmitted']
plt.figure(figsize=(80 / 25.4, 80 / 25.4))
(data.groupby('channel_width')['track_end'].var()/ 300).plot(style='o-')
plt.xlabel('channel width')
plt.xlim(left=0)
plt.ylim(bottom=0, top = 3)
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
plt.ylabel(r'variance [min$^{2}$/step]')
plt.savefig(Path(__file__).parent.parent / 'panels' / 'figS1_plot-3.png')
plt.savefig(Path(__file__).parent.parent / 'panels' / 'figS1_plot-3.svg')