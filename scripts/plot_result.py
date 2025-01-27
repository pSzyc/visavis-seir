# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024-2025 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk

    
import pandas as pd
import matplotlib.pyplot as plt
from subplots_from_axsize import subplots_from_axsize


def plot_result(result, outfile=None, title=None, t_min=None, t_max=None, ax=None, panel_size=(20, 8), show=True, **kwargs):
    return plot_result_from_states(result.states, outfile=outfile, title=title, t_min=t_min, t_max=t_max, ax=ax, panel_size=panel_size, show=show, **kwargs)


def plot_result_from_states(data, outfile=None, title=None, t_min=None, t_max=None, ax:plt.Axes = None, panel_size=(20, 8), show=True, **kwargs):
    
    
    # act = 1.0 * ((data['E'] > 0) | (data['I'] > 0))
    act = pd.Series(
        (data['E'].to_numpy() + data['I'].to_numpy() > 0) * 1.,
        index = data.index,
    )

    activity = act.groupby(['seconds', 'h']).mean().unstack()

    return plot_result_from_activity(activity, outfile=outfile, title=title, t_min=t_min, t_max=t_max, ax=ax, panel_size=panel_size, show=show, **kwargs)


def plot_result_from_activity(activity, outfile=None, title=None, t_min=None, t_max=None, transpose=False, ax:plt.Axes = None, panel_size=(20, 8), show=True, **kwargs,):


    if ax is None:
        fig, ax = subplots_from_axsize(
            1, 1, axsize=panel_size,
            left=0., right=0., bottom=0., top=0.
        )
    else:
        fig = ax.get_figure()


    if t_min is None:
        t_min = activity.index.get_level_values('seconds').min()
    if t_max is None:
        t_max = activity.index.get_level_values('seconds').max()

    img = activity[[t_min <= x <= t_max for x in activity.index.get_level_values('seconds')]].to_numpy().T
    
    ax.imshow(
        img.T if transpose else img,
        **{
            'cmap': 'grey',
            'origin': 'upper' if transpose else 'lower',
            'aspect': 'auto',
            'interpolation': 'none',
            **kwargs
        }
    )

    ax.set_xlabel('position along channel' if transpose else 'time')
    ax.set_axis_off()
    
    if title is not None:
        ax.annotate(
            title, (0.5, 0.9),
            xycoords='axes fraction', va='center', ha='center',
            color='red', fontsize=32
        )

    if outfile is not None:
        fig.savefig(outfile)
        plt.close(fig)
    elif show:
        plt.show()


    return fig, ax
