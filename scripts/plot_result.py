import pandas as pd
import matplotlib.pyplot as plt
from subplots_from_axsize import subplots_from_axsize


def plot_result(result, outfile=None, title=None, t_min=None, t_max=None, ax=None, panel_size=(20, 8), show=True):
    return plot_result_from_states(result.states, outfile=outfile, title=title, t_min=t_min, t_max=t_max, ax=ax, panel_size=panel_size, show=show)


def plot_result_from_states(data, outfile=None, title=None, t_min=None, t_max=None, ax:plt.Axes = None, panel_size=(20, 8), show=True):
    
    
    # act = 1.0 * ((data['E'] > 0) | (data['I'] > 0))
    act = pd.Series(
        (data['E'].to_numpy() + data['I'].to_numpy() > 0) * 1.,
        index = data.index,
    )

    activity = act.groupby(['seconds', 'h']).mean().unstack()

    return plot_result_from_activity(activity, outfile=outfile, title=title, t_min=t_min, t_max=t_max, ax=ax, panel_size=panel_size, show=show)


def plot_result_from_activity(activity, outfile=None, title=None, t_min=None, t_max=None, ax:plt.Axes = None, panel_size=(20, 8), show=True):


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

    img = activity[activity.index.get_level_values('seconds').between(t_min, t_max)].to_numpy().T
    
    ax.imshow(
        img,
        cmap='gray',
        origin='lower',
        aspect='auto',
        interpolation='none',
    )

    ax.set_xlabel('time')
    ax.set_axis_off()
    
    if title is not None:
        ax.annotate(
            title, (0.5, 0.9),
            xycoords='axes fraction', va='center', ha='center',
            color='red', fontsize=32
        )

    if outfile is not None:
        print("Don't call me off")
        fig.savefig(outfile)
        print("Please!")
        plt.close(fig)
    elif show:
        plt.show()


    return fig, ax
