import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma
from subplots_from_axsize import subplots_from_axsize
from pathlib import Path

import sys
root_repo_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py
from scripts.style import *
plt.rcParams['svg.fonttype'] = 'path'

panel_dir = Path(__file__).parent.parent / 'panels'


def erlang_pdf(k, rate):
    def pdf(x):
        return rate**k * x**(k-1) * np.exp(-rate * x) / gamma(k)
    return pdf


xx = np.linspace(0,100,1001)
fig, axs = subplots_from_axsize(1, 3, (1.2, .7), left=.2, bottom=.3, wspace=[.3, .3])


for ax, (letter, k, rate, color, sigma_label_loc) in zip(axs, [
    ('E', 4, 1, 'goldenrod', 'upper'),
    ('I', 2, 1, 'firebrick', 'upper'),
    ('R', 4, 1/15, 'mediumturquoise', 'lower'),
]):
    pdf = erlang_pdf(k, rate)(xx)
    ax.plot(xx, pdf, color=color)
    ax.set_xticks([k / rate])
    # ax.set_xticks([0, 50, 100], minor=True)
    ax.xaxis.set_major_formatter((f"$\\tau_{letter}$ = " "{:.0f} min").format)
    ax.set_yticks([])
    ax.spines[['top', 'left', 'right', 'bottom']].set_color('gray')
    # ax.set_title("time spent in state $R$")
    ypos = 1.2*pdf.max() / 2 if sigma_label_loc == 'upper' else .4*pdf.max() / 2
    sigma = np.sqrt(k) / rate
    ax.annotate(f'$\\sigma_{letter}\\ = \\ \\tau_{letter}\\ / \\sqrt{{n_{letter}}} = {sigma:.2g}$ min', ((k - .5)/rate - 1.5*sigma + 10, ypos), )
    ax.annotate('', 
        xy=((k - .5)/rate - sigma, pdf.max() / 2), 
        xytext=((k - .5)/rate + sigma, pdf.max() / 2), 
        arrowprops={
            'arrowstyle': '|-|',
            'shrinkA': 0,
            'shrinkB': 0,
            'mutation_scale': 2.5,
        })
    ax.annotate('', 
        xy=((k - .5)/rate, pdf.max() / 2), 
        xytext=((k - .5)/rate + sigma, pdf.max() / 2), 
        arrowprops={
            'arrowstyle': '|-|',
            'shrinkA': 0,
            'shrinkB': 0,
            'mutation_scale': 1.25,
        })
    if sigma_label_loc == 'lower':
        ax.annotate('', 
            xy=((k - .5)/rate - sigma, .9 * (pdf.max() / 2)), 
            xytext=((k - .5)/rate - sigma / 2 , .65 * (pdf.max() / 2)), 
            arrowprops={
                'shrinkA': 0,
                'shrinkB': 0,
                'arrowstyle': '-',
                'ls': ':',
            })
        ax.annotate('', 
            xy=((k - .5)/rate, .9 * (pdf.max() / 2)), 
            xytext=((k - .5)/rate - sigma / 2 , .65 * (pdf.max() / 2)), 
            arrowprops={
                'shrinkA': 0,
                'shrinkB': 0,
                'arrowstyle': '-',
                'ls': ':',
            })

plt.savefig(panel_dir / 'fig1B.png')
plt.savefig(panel_dir / 'fig1B.svg')
    