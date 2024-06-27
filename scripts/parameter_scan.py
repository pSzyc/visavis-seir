# Script used in Information transmission in a cell monolayer: A numerical study, Nałęcz-Jawecki et al. 2024
# Copyright (C) 2024 Paweł Nałęcz-Jawecki, Przemysław Szyc, Frederic Grabowski, Marek Kochańczyk
# Available under GPLv3 licence, see README for more details
    
from subplots_from_axsize import subplots_from_axsize
from matplotlib.ticker import MultipleLocator
from .defaults import PARAMETERS_DEFAULT

def plot_parameter_scan(feature_rates, feature_states, feature_name, logscale=False, ylim=None, grid=False, axs=None, plot_rates=True, plot_states=True, **kwargs):
    if axs is None:
        fig, axs = subplots_from_axsize(plot_rates + plot_states, 4, (1.38, 1.1), wspace=.2, hspace=.4)
    
    axs = axs.reshape(-1, 4)
    states_row = 1 if plot_rates else 0
    first_states_col = 1 if plot_rates else 0


    if plot_rates:

        ax = axs[0,0]
        altered_parameter = 'c_rate'
        feature_values = feature_rates.loc[altered_parameter]
        ax.plot(
            1 / (feature_values.index.get_level_values('fold_change') * PARAMETERS_DEFAULT[altered_parameter]), 
            feature_values, 
            'o-',
            **{
                    'label': feature_name,
                    'color': 'navy',
                    'ms': 3,
                    **kwargs
                }
            )

        ax.plot(
            1 / PARAMETERS_DEFAULT[altered_parameter], 
            feature_rates.loc[altered_parameter, 1], 
            'o',
            fillstyle='none',
            **{
                    'color': 'navy',
                    **kwargs
            }
            )

        ax.set_xlabel(r"$\tau_{\mathrm {act}}$ [min]")
        ax.set_ylabel(feature_name)
        if grid:
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.grid(which='major', ls=':', alpha=.4)


        for ax, species in zip(axs[0][1:], ['e', 'i', 'r']):
            altered_parameter = f'{species}_forward_rate'
            corresponding_states = f'{species}_subcompartments_count'
            feature_values = feature_rates.loc[altered_parameter]
            ax.plot(
                PARAMETERS_DEFAULT[corresponding_states] / (feature_values.index.get_level_values('fold_change') * PARAMETERS_DEFAULT[altered_parameter]), 
                feature_values, 
                'o-',
                **{
                    'color': 'navy',
                    'ms': 3,
                    **kwargs
                }
                )

            ax.plot(
                PARAMETERS_DEFAULT[corresponding_states] / PARAMETERS_DEFAULT[altered_parameter], 
                feature_rates.loc[altered_parameter, 1.], 
                'o',
                fillstyle='none',
                **{
                        'color': 'navy',
                        **kwargs,
                }
                )
            ax.set_xlabel(r"$\tau_{\mathrm {" + species.upper() + r"}}$ [min]")
            ax.tick_params(labelleft=False)
            ax.sharey(axs[0,0])
            if grid:
                ax.yaxis.set_minor_locator(MultipleLocator(1))
                ax.yaxis.grid(which='major', ls=':', alpha=.4)
                




    if plot_states:
        if plot_rates:
            axs[states_row,0].set_visible(False)
        else:
            axs[states_row,-1].set_visible(False)


        for ax, species in zip(axs[states_row][first_states_col:axs.shape[-1]+first_states_col-1], ['e', 'i', 'r']):
            altered_parameter = f'{species}_subcompartments_count'
            # corresponding_states = f'{species}_subcompartments_count'
            feature_values = feature_states.loc[altered_parameter]
            ax.plot(
                feature_values.index.get_level_values('n_states'), 
                feature_values, 
                'o-',
                **{
                    'color': 'navy',
                    'ms': 3,
                    **kwargs
                }
                )

            ax.plot(
                PARAMETERS_DEFAULT[altered_parameter],
                feature_states.loc[altered_parameter, PARAMETERS_DEFAULT[altered_parameter]], 
                'o',
                fillstyle='none',
                **{
                        'color': 'navy',
                        **kwargs
                }
                )
            ax.set_xlabel(r"$n_{\mathrm {" + species.upper() + r"}}$")
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(labelleft=False)
            ax.sharey(axs[0,0])
            if grid:
                ax.yaxis.set_major_locator(MultipleLocator(2))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
                ax.yaxis.grid(which='major', ls=':', alpha=.4)


        axs[states_row, first_states_col].set_ylabel(feature_name)
        axs[states_row, first_states_col].tick_params(labelleft=True)

    if logscale:
        for ax in axs.flatten():
            ax.set_yscale('log')
    if ylim:
        ax.set_ylim(ylim)



    return axs
