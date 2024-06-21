from matplotlib.legend_handler import HandlerTuple, HandlerBase
from itertools import cycle, product
import numpy as np
from matplotlib.patches import Rectangle

class HandlerTupleVertical(HandlerBase):
    """
    Handler for Tuple.
    """

    def __init__(self, nrows=None, ncols=None, hpad=None, vpad=0, **kwargs):
        """
        Parameters
        ----------
        nrows : int or None, default: None
            The number of cols to divide the legend area into. 
            If None, use (the ceil of) the length 
            of the input tuple divided by ncols.
            If both nrows and ncols are None, use ncols=1 
            and nrows equal to the length of the input tuple.
        ncols : int or None, default: None
            The number of rows to divide the legend area into. 
            If None, use (the ceil of) the length 
            of the input tuple divided by nrows.
        hpad : float, default: :rc:`legend.borderpad`
            Horizontal padding in units of fraction of font size.
        vpad : float, default: :rc:`legend.borderpad`
            Vertical padding in units of fraction of font size.
        **kwargs
            Keyword arguments forwarded to `.HandlerBase`.
        """
        self._nrows = nrows
        self._ncols = ncols
        self._hpad = hpad
        self._vpad = vpad
        super().__init__(**kwargs)

    

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # docstring inherited
        handler_map = legend.get_legend_handler_map()

        if self._nrows is None and self._ncols is None:
            ncols = 1
            nrows = len(orig_handle)
        elif self._nrows is None:
            ncols = self._ncols
            nrows = (len(orig_handle) - 1) // ncols + 1
        elif self._ncols is None:
            nrows = self._nrows
            ncols = (len(orig_handle) - 1) // nrows + 1
        else:
            ncols = self._ncols
            nrows = self._nrows


        if self._hpad is None:
            hpad = legend.borderpad * fontsize
        else:
            hpad = self._hpad * fontsize

        if self._vpad is None:
            vpad = legend.borderpad * fontsize
        else:
            vpad = self._vpad * fontsize

        a_list = []
        # a_list = [Rectangle((xdescent, ydescent), width, height, color='k', alpha=.3)]

        total_width = width
        total_height = height

        if ncols > 1:
            width = (width - hpad * (ncols - 1)) / ncols

        if nrows > 1:
            height = (height - vpad * (nrows - 1)) / nrows

        xyds_cycle = cycle(
            ((xdescent - (width + hpad) * it_col), (ydescent - total_height + 2 * (height + vpad) * it_row)) 
            for it_row, it_col in product(range(nrows), range(ncols))
            )
        
        xyds_cycle = cycle(
            ((xdescent + offset_x), (ydescent + offset_y)) 
            for offset_y, offset_x in product(np.linspace(0., total_height, nrows) - 0 * total_height, - np.linspace(0., total_width, ncols, endpoint=False))
            )
        

        # a_list = []
        for it,handle1 in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle1)
            xy=next(xyds_cycle)
            _a_list = handler.create_artists(
                legend, handle1,
                *xy, width, height, fontsize, trans)
            a_list.extend(_a_list)
            # a_list.extend([Rectangle(xy, width, height, color=f'C{it}', alpha=.3)])

        

        return a_list
