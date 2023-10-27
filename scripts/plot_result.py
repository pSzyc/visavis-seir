import pandas as pd
import matplotlib.pyplot as plt
from subplots_from_axsize import subplots_from_axsize


def plot_result(result, outfile=None, title=None, t_min=None, t_max=None, ax=None, panel_size=(20, 8), show=True):
    return plot_result_from_states(result.states, outfile=outfile, title=title, t_min=t_min, t_max=t_max, ax=ax, panel_size=panel_size, show=show)


def plot_result_from_states(data, outfile=None, title=None, t_min=None, t_max=None, ax:plt.Axes = None, panel_size=(20, 8), show=True):
    
    if ax is None:
        fig, ax = subplots_from_axsize(
            1, 1, axsize=panel_size,
            left=0., right=0., bottom=0., top=0.
        )
    else:
        fig = ax.get_figure()

    if t_min is None:
        t_min = data['seconds'].min()
    if t_max is None:
        t_max = data['seconds'].max()

    data_selected = data[data['seconds'].between(t_min, t_max)].set_index(['h', 'seconds'])
    
    # act = 1.0 * ((data['E'] > 0) | (data['I'] > 0))
    act = pd.Series(
        (data_selected['E'].to_numpy() + data_selected['I'].to_numpy() > 0) * 1.,
        index = data_selected.index,
    )
    del data_selected
    
    img = act.groupby(['h', 'seconds']).mean().unstack().to_numpy()
    
    # data_selected['E'] = data_selected['E'] > 0
    # data_selected['I'] = data_selected['I'] > 0
    # data_selected['R'] = data_selected['R'] > 0

    # img_E = data_selected.groupby(['seconds', 'h'])['E'].mean().unstack().to_numpy().T
    # img_I = data_selected.groupby(['seconds', 'h'])['I'].mean().unstack().to_numpy().T
    # img_R = data_selected.groupby(['seconds', 'h'])['R'].mean().unstack().to_numpy().T

    # img = img_E + img_I

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
