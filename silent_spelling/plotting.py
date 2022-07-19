# -*- coding: utf-8 -*-
"""
A module containing utility functionality for figure generation.

:Author: David Moses
:Copyright: Copyright (c) 2021, David Moses, All rights reserved.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def empty_dict():
    """
    Returns an empty `SemiFrozenDict`.

    This function is useful for creating an empty dictionary default argument
    value for other functions.

    Returns
    -------
    SemiFrozenDict
        An empty dictionary instance.
    """

    return SemiFrozenDict(())


class SemiFrozenDict(dict):
    """
    A dictionary sub-class that can be used to create frozen dictionaries.

    An instance of this class cannot have items added to it, and existing keys
    cannot be re-assigned to new values. However, if a value in an instance of
    this class is mutable, it is possible for that value to be modified.

    Note
    ----
    Creating a copy of an instance of this class using the `copy` method will
    return a `dict` object (not another instance of this class).
    """

    def __setitem__(self, key, value):
        """
        Raises a `TypeError` if item assignment is attempted.

        Raises
        ------
        TypeError
            Disables item assignment for instances of this class by raising
            this exception if item assignment is attempted.
        """

        raise TypeError("'{}' object does not support item assignment".format(
            self.__class__.__name__
        ))


def plot_images_and_elecs(
        elec_loc_file_path, all_image_params, elec_size_color_params,
        elec_weights=None, plot_colorbar=False, x_shift=0., y_shift=0.,
        x_scale=1., y_scale=1., add_width=False, add_height=False,
        elec_plot_params=empty_dict(), figure_params=empty_dict(),
        apply_formatting=True, colorbar_title='',
        colorbar_orientation='horizontal', fig=None, ax=None,
        ax_cbar=None, show_fig=True
):
    """
    Plots one or more images with overlaid electrode information.

    Parameters
    ----------
    elec_loc_file_path : str
        The file path at which the electrode coordinate locations are stored.
        This file should be loadable using `numpy.load`, and should contain a
        float array of shape `(num_electrodes, 2)` containing the X-Y
        coordinates for each electrode. These coordinates should correspond to
        the pixels contained in the images.
    all_image_params : list or tuple or dict
        A list or tuple in which each element is a dictionary containing
        keyword arguments to pass to the `load_asset_image` function to load
        the desired image data. If this is a dictionary instance instead, it is
        treated as if it were a 1-element list containing the provided
        dictionary.
    elec_size_color_params : dict
        A dictionary of keyword arguments to pass to the
        `get_elec_sizes_and_colors` function to load the electrode size and
        color information for the plot. The `weights` keyword argument for that
        function will be determined from the `elec_weights` argument to this
        function, overridding the value from this dictionary if one was
        provided.
    elec_weights : ndarray or None
        Electrode weights, which should be a float array of shape
        `(num_electrodes,)`. If this is `None`, an array of the appropriate
        shape will be created (informed by the loaded electrode coordinate
        locations) containing flat (equal) values.
    plot_colorbar : bool
        Specifies whether or not to include a colorbar plot. If the value
        mapped to the `"color_spec"` key in `elec_size_color_params` is not a
        valid colormap or a string specifying a colormap, then a colorbar will
        not be plotted (but, if this argument is `True`, an empty space in the
        figure will still be allocated to where the colorbar would have been).
        If "relative" colors are specified in  `elec_size_color_params`, then
        the full colormap will be plotted. Otherwise, a subset of the colormap
        from `max(0, min(elec_weights))` to `min(1, max(elec_weights))` will
        be plotted.
    x_shift : float
        The shift to apply (add) to each electrode X coordinate.
    y_shift : float
        The shift to apply (add) to each electrode Y coordinate.
    x_scale : float
        The scale to apply (multiply) to each electrode X coordinate.
    y_scale : float
        The scale to apply (multiply) to each electrode Y coordinate.
    add_width : bool
        Specifies whether or not to add the width of the first image to the
        electrode X coordinates, which can be useful if using an `x_scale` of
        `-1` to invert the plotted electrode X locations. If this evaluates to
        `True`, this transformation will be applied after applying the scale
        and shift.
    add_height : bool
        Specifies whether or not to add the height of the first image to the
        electrode Y coordinates, which can be useful if using an `y_scale` of
        `-1` to invert the plotted electrode Y locations. If this evaluates to
        `True`, this transformation will be applied after applying the scale
        and shift.
    elec_plot_params : dict
        Keyword arguments to pass to the `scatter` method used to plot the
        electrodes in the image. The `x`, `y`, `s`, and `color` keyword
        arguments are specified within this function, but they will be
        overridden by items in this dictionary if there is a conflict.
    figure_params : dict
        Keyword arguments to pass to the `pyplot.figure` function when creating
        the figure. This is only used if `fig` is `None`.
    apply_formatting : bool
        Specifies whether or not to format the plot by removing the axis
        borders and restricting the size to the image dimensions.
    colorbar_title : str
        The title to apply to the colorbar. This is only used if
        `plot_colorbar` evaluates to `True`.
    colorbar_orientation : str
        The orientation of the colorbar. This is only used if `plot_colorbar`
        evaluates to `True`.
    fig : Figure or None
        The `Figure` instance to use. If this is `None`, a new figure will be
        created.
    ax : AxesSubplot or None
        The `AxesSubplot` instance to use to plot the images and electrodes.
        If this is `None`, a new subplot will be created.
    ax_cbar : AxesSubplot or None
        The `AxesSubplot` instance to use to plot color bar (if one is
        desired). If this is `None`, a new subplot will be created for the
        colorbar (if desired).
    show_fig : bool
        Specifies whether or not to show the figure after plotting.
    """

    # Converts a single image parameter dictionary into a 1-element list
    # containing the dictionary as the only element
    if isinstance(all_image_params, dict):
        all_image_params = [all_image_params]

    # Loads all of the image data
    all_image_data = [
        load_asset_image(**cur_params) for cur_params in all_image_params
    ]

    # Loads the electrode location coordinates and applies the desired
    # transformations
    elec_locs = np.load(get_asset_file_path(elec_loc_file_path))

    image_height, image_width = all_image_data[0].shape[:2]
    elec_locs[:, 0] = (elec_locs[:, 0] * x_scale) + x_shift
    elec_locs[:, 1] = (elec_locs[:, 1] * y_scale) + y_shift

    if add_width:
        elec_locs[:, 0] += image_width
    if add_height:
        elec_locs[:, 1] += image_height

    # Creates the figure (if one was not provided)
    if fig is None:
        fig = plt.figure(**figure_params)

    # Creates the image axis (if one was not provided) and, if desired, the
    # colorbar axis (if one was not provided)
    gridspec = fig.add_gridspec(20, 1)

    if ax is None:
        if plot_colorbar:
            ax = fig.add_subplot(gridspec[:18, :])
        else:
            ax = fig.add_subplot(111)

    if plot_colorbar and ax_cbar is None:
        ax_cbar = fig.add_subplot(gridspec[-1, :])

    # Plots the images
    for cur_image_data in all_image_data:
        ax.imshow(cur_image_data)

    # If weights were not provided, default flat weights are used
    if elec_weights is None:
        elec_weights = np.ones(elec_locs.shape[0], dtype=float)
        elec_weights /= np.sum(elec_weights)

    # Computes the electrode sizes and colors
    cur_sizes, cur_colors = get_elec_sizes_and_colors(
        weights=elec_weights, **elec_size_color_params
    )

    # Plots the electrodes
    actual_elec_plot_params = dict(
        x=elec_locs[:, 0], y=elec_locs[:, 1], s=cur_sizes, color=cur_colors,
    )
    actual_elec_plot_params.update(elec_plot_params)
    ax.scatter(**actual_elec_plot_params)

    # Formats the plot (if desired)
    if apply_formatting:
        ax.axis('off')
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)

    # Adds a colorbar (if appropriate)
    if plot_colorbar:
        try:
            cmap = elec_size_color_params['color_spec']
            if not isinstance(cmap, mpl.colors.Colormap):
                cmap = plt.get_cmap(cmap)
        except ValueError:
            ax_cbar.axis('off')
        else:

            min_val = np.min(elec_weights)
            max_val = np.max(elec_weights)

            if not elec_size_color_params.get('color_params', {}).get(
                    'relative', False
            ):
                min_val = max(0., min_val)
                max_val = min(1., max_val)

                cmap = mpl.colors.LinearSegmentedColormap.from_list(
                    name='subset_cmap',
                    colors=cmap(np.linspace(min_val, max_val, 1000))
                )

            colorbar = mpl.colorbar.ColorbarBase(
                ax=ax_cbar, cmap=cmap, orientation=colorbar_orientation
            )
            colorbar.set_label(colorbar_title)

            colorbar.set_ticks(np.linspace(0., 1., 5, endpoint=True))
            colorbar.set_ticklabels(
                ['{:g}'.format(i) for i in
                 np.linspace(min_val, max_val, 5, endpoint=True)]
            )

    # Shows the figure (if desired)
    if show_fig:
        plt.show()


def load_asset_image(file_name, invert_y=False, alpha=None,
                     only_apply_alpha_to_nonzero=False,
                     mask_pixel_values=None):
    """
    Loads, processes, and returns pixel data associated with an asset image.

    Parameters
    ----------
    file_name : str
        The asset file name. 
    invert_y : bool
        Specifies whether or not to invert the image along the y-axis, which
        can be useful when preparing the image data for a plot. Note that this
        will invert the first dimension of the pixel data.
    alpha : float or None
        Either a float in the range `[0, 1]` specifying the alpha value to
        apply to the pixels or `None`. If this is `None`, the image alpha
        values are not modified.
    only_apply_alpha_to_nonzero : bool
        Specifies whether or not to apply the alpha values to pixels that have
        a non-zero alpha value.
    mask_pixel_values : array-like or None
        If this is not `None`, it is assumed that the specified file is a
        `numpy` data file containing a 2-D boolean array. This array specifies
        which pixels are "inactive" and which are "active". A float array of
        shape `(x.shape[0], x.shape[1], 4)` will be created to represent the
        pixel data, where `x` is the boolean array loaded from the file. This
        new float array will contain the pixel data for the image. The two
        elements in this argument value should be the data for each "inactive"
        pixel and the data for each "active" pixel (in that order). The
        `matplotlib.colors.to_rgba` function will be used to convert the
        value of each element to the RGBA pixel values. If this argument value
        is `None`, it is assumed that the specified file is an image file, and
        the `matplotlib.image.imread` function is used to load the pixel data.

    Returns
    -------
    ndarray
        An array of shape `(height, width, 4)` containing RGBA values for each
        pixel of the image.
    """

    # If mask pixel values were not provided, the image is loaded directly
    if mask_pixel_values is None:
        image_data = mpimg.imread(get_asset_file_path(file_name))

    # Otherwise, additional steps are taken to compute the image pixel values
    else:

        # Extracts the "active" and "inactive" pixel colors
        inactive_pixel_color, active_pixel_color = [
            mpl.colors.to_rgba(c) for c in mask_pixel_values
        ]

        # Loads the pixel boolean mask
        image_mask = np.load(get_asset_file_path(file_name))

        # Creates the image pixel array
        image_data = np.empty(shape=(image_mask.shape + (4,)), dtype=float)
        image_data[~image_mask] = inactive_pixel_color
        image_data[image_mask] = active_pixel_color

    # Inverts the y-axis (if desired)
    if invert_y:
        image_data = image_data[::-1, :, :]

    # If the desired alpha value is non-zero, additional steps are taken
    if alpha is not None:

        # If desired, the alpha value will only be applied to pixels with
        # non-zero alpha values
        if only_apply_alpha_to_nonzero:
            nonzero_inds = np.nonzero(image_data[:, :, -1])
            target_alpha_inds = nonzero_inds + (
                np.full(shape=nonzero_inds[0].shape, dtype=int, fill_value=-1),
            )
            image_data[target_alpha_inds] = alpha

        # Otherwise, the alpha value is applied to all pixels
        else:
            image_data[:, :, -1] = alpha

    # Returns the image data
    return image_data


def get_elec_sizes_and_colors(
        weights, color_spec, size_params=empty_dict(),
        color_params=empty_dict(), alpha_params=empty_dict()
):
    """
    Computes electrode sizes and RGBA colors from weights values.

    Electrode sizes, colors, and alpha values are controlled by the weights
    and function parameters. These values are processed in the following order:
    1) Compute initial values from the weights (direct or relative).
    2) Shift the values.
    3) Scale the values.
    4) Exponentiate the values.
    5) Clip the values.

    If a colormap is specified, colors will be scaled such
    Parameters
    ----------
    weights : array-like
        A 1D array of shape `(num_electrodes,)` containing electrode weights
        (as floats). All values should lie in the range of `[0, 1]`.
    color_spec : object
        Either an object recognizable by `matplotlib` as a color or a color
        map (or string specifying a color map). If it is a single color,
        it will be used for all electrodes. Otherwise, the provided weights
        will be used to obtain the color from the color map. These colors may
        still be altered using new alpha values.
    size_params : dict
        A dictionary that may contain any of the following items (the presence
        of one of these items in this dictionary argument will override the
        corresponding default value):
        - "min" : float
            The minimum electrode size. Default is `0.0`.
        - "max" : float
            The maximum electrode size. Default is `+inf`.
        - "relative" : bool
            Specifies whether or not to scale the sizes relative to the range
            of `weight` values (which would cause the electrode with the
            smallest weight to have a size of `0.0` and the electrode with the
            largest weight to have a size of `1.0`). Otherwise, the weights
            will be used as the sizes. All sizes are still shifted, scaled,
            exponentiated, and clipped before they are returned. Default is
            `True`.
        - "shift" : float
            All electrode sizes will be shifted (incremented) by this value.
            Default is `0.0`.
        - "scale" : float
            All electrode sizes will be scaled (multiplied) by this value.
            Default is `1.0`.
        - "exponent" : float
            All electrode sizes will be exponentiated by this value. Default is
            `1.0`.
    color_params : dict
        A dictionary that resembles the `size_params` dictionary except that it
        modifies the color values. Items in this dictionary will only be used
        if a color map was specified in `color_spec` (and not just a single
        color).
    alpha_params : dict
        A dictionary that resembles the `size_params` dictionary except that it
        modifies the alpha values.

    Returns
    -------
    sizes : array-like
        A 1-D float array of shape `(num_electrodes,)` containing electrode
        sizes.
    colors : array-like
        A 2-D float array of shape `(num_electrodes, 4)` electrode RGBA values.
    """

    # Defines a local dictionary from which default values will be loaded for
    # any missing parameters
    _defaults = {
        "min"     : 0.,
        "max"     : float('inf'),
        "relative": True,
        "shift"   : 0.,
        "scale"   : 1.,
        "exponent": 1.
    }

    # Defines a local function to use to obtain parameter values
    def _get_spec(_cur_params, _cur_spec):
        return _cur_params.get(_cur_spec, _defaults[_cur_spec])

    # Determines the number of electrodes
    num_electrodes = len(weights)

    # Computes the minimum and maximum weight and the relative weights
    weights = np.asarray(weights)
    if np.all(weights == weights[0]):
        relative_weights = weights / np.sum(weights)
    else:
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        relative_weights = (weights - min_weight) / (max_weight - min_weight)

    # Iterates through the electrode sizes, colors, and alpha values
    elec_params = {}
    for cur_key, cur_params in zip(
            ('sizes', 'colors', 'alphas'),
            (size_params, color_params, alpha_params)
    ):

        # Obtains either the relative weights or original weights
        if _get_spec(cur_params, 'relative'):
            cur_vals = relative_weights.copy()
        else:
            cur_vals = weights.copy()

        # Applies operations to the values
        cur_vals = np.clip(
            (((cur_vals + _get_spec(cur_params, 'shift'))
              * _get_spec(cur_params, 'scale'))
             ** _get_spec(cur_params, 'exponent')),
            _get_spec(cur_params, 'min'), _get_spec(cur_params, 'max')
        )

        # Stores the current values
        elec_params[cur_key] = cur_vals

    # Initializes the color array, either using a static color or a colormap
    # (depending on what was provided)
    try:
        colors = mpl.colors.to_rgba_array([color_spec] * num_electrodes)
    except ValueError:
        if not isinstance(color_spec, mpl.colors.Colormap):
            color_spec = plt.get_cmap(color_spec)
        colors = color_spec(elec_params['colors'])

    # Adds the alpha values to the color array
    colors[:, -1] = elec_params['alphas']

    # Returns the sizes and colors
    return (elec_params['sizes'], colors)


def get_asset_file_path(asset_name):
    """
    Returns the absolute file path to an asset.
    
    Note that this function does not determine whether or not the desired file
    actually exists.

    Parameters
    ----------
    asset_name : str
        The file name of the asset.
    
    Returns
    -------
    str
        The absolute file path to an asset.
    """

    return f'recon/{asset_name}'
