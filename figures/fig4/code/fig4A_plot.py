import pandas as pd
from matplotlib import pyplot as plt
import shutil

from pathlib import Path
from subplots_from_axsize import subplots_from_axsize

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.analyze_binary import plot_scan

data_dir = Path(__file__).parent.parent / 'data'
panels_dir = Path(__file__).parent.parent / 'panels'
panels_dir.mkdir(parents=True, exist_ok=True)

for fields in 'c', 'rl', 'cm', 'cp', 'cmp':
    for k_neighbors in (15, 25):
        for reconstruction in (True, False):

            suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"

            entropies = pd.read_csv(data_dir / f'fig4A_entropies2{suffix}.csv')

            fig, ax = plot_scan(
                entropies, 
                c_field='channel_length',
                x_field='interval',
                y_field='bitrate_per_hour',
            )

            ax.set_ylim(0,1)

            fig.savefig(panels_dir / f'fig4A{suffix}.svg')
            fig.savefig(panels_dir / f'fig4A{suffix}.png')
            plt.close(fig)

fig, ax = subplots_from_axsize(1, 1, (3, 2.5), left=0.7)
 
entropies = pd.read_csv(data_dir / f'fig4A_entropies2-c25.csv')

plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    ms=3,
    ax=ax,
)

entropies = pd.read_csv(data_dir / f'fig4A_entropies2-cm15.csv')

plot_scan(
    entropies, 
    c_field='channel_length',
    x_field='interval',
    y_field='bitrate_per_hour',
    alpha=0.3,
    ms=3,
    ax=ax,
)

ax.set_ylim(0,1)
handles = ax.get_legend().legend_handles
ax.legend(handles=handles[:len(handles)//2])

fig.savefig(panels_dir / f'fig4A.svg')
fig.savefig(panels_dir / f'fig4A.png')
plt.close(fig)


# shutil.copy2(panels_dir / 'fig4A-c25.svg', panels_dir / 'fig4A.svg')
# shutil.copy2(panels_dir / 'fig4A-c25.png', panels_dir / 'fig4A.png')
(panels_dir / '../../fig5/panels').mkdir(parents=True, exist_ok=True)
shutil.copy2(panels_dir / 'fig4A-c25-reconstruction.svg', panels_dir / '../../fig5/panels/fig5A.svg')
shutil.copy2(panels_dir / 'fig4A-c25-reconstruction.png', panels_dir / '../../fig5/panels/fig5A.png')



for fields in 'c', 'rl':
    for k_neighbors in (15, 25):
        for reconstruction in (True, False):

            suffix = f"-{fields}{k_neighbors}{'-reconstruction' if reconstruction else ''}"

            entropies = pd.read_csv(data_dir / f'fig4A_entropies2{suffix}.csv')

            fig, ax = plot_scan(
                entropies, 
                c_field='channel_length',
                x_field='interval',
                y_field='efficiency',
            )

            ax.set_ylabel('efficiency')
            ax.set_ylim(0,1.02)

            fig.savefig(panels_dir / f'figS4-1{suffix}.svg')
            fig.savefig(panels_dir / f'figS4-1{suffix}.png')
            plt.close(fig)


(panels_dir / '../../figS4-1/panels').mkdir(parents=True, exist_ok=True)

shutil.copy2(panels_dir / 'figS4-1-c25.svg', panels_dir / '../../figS4-1/panels/figS4-1.svg')
shutil.copy2(panels_dir / 'figS4-1-c25.png', panels_dir / '../../figS4-1/panels/figS4-1.png')
