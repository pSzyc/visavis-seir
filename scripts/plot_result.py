import matplotlib.pyplot as plt
from subplots_from_axsize import subplots_from_axsize


def plot_result(result, outfile=None, title=None, t_min=None, t_max=None):
    data = result.states
    
    fig, ax = subplots_from_axsize(
        1, 1, axsize=(20, 8),
        left=0., right=0., bottom=0., top=0.
    )

    data_selected = data.copy()
    
    if t_min is not None:
        data_selected =  data_selected[data_selected['seconds'] >= t_min].copy()
        
    if t_max is not None:
        data_selected =  data_selected[data_selected['seconds'] <= t_max].copy()
    
    
    data_selected['E'] = data_selected['E'] > 0
    data_selected['I'] = data_selected['I'] > 0
    data_selected['R'] = data_selected['R'] > 0

    img_E = data_selected.groupby(['seconds', 'h'])['E'].mean().unstack().to_numpy().T
    img_I = data_selected.groupby(['seconds', 'h'])['I'].mean().unstack().to_numpy().T
    img_R = data_selected.groupby(['seconds', 'h'])['R'].mean().unstack().to_numpy().T

    img = img_E + img_I

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
        fig.savefig(outfile)
        plt.close(fig)
    else:
        plt.show()
