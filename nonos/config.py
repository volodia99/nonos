DEFAULTS = {
    # directory where the simulation results are stored
    "datadir": ".",
    # which field?
    "field": "RHO",
    # 1D axisymmetric porfile or 2D map
    "dimensionality": 2,
    # a single output number ('on') to display or a range (min, max, [step])
    "on": "unset",
    # perturbation of field
    "diff": False,
    # field in log
    "log": False,
    # default extent of the matplotlib window
    "range": "unset",
    # default min field when diff
    "vmin": "unset",
    # default max field when diff
    "vmax": "unset",
    # plane to represent (default is midplane)
    "plane": "xy",
    # do we perform a slice in the 3rd dimension, or an average?
    "slice": False,
    # do we compute lic streamlines?
    "lic": "unset",
    # select interpolation cell refinement
    "licres": 5,
    # do we display the progress (loading+plotting)
    "progressBar": False,
    # do we display the grid?
    "grid": False,
    # is there a planet in the grid?
    "isPlanet": False,
    # do the grid rotate with the planet?
    "corotate": False,
    # number of cpus to use
    "ncpu": 1,
    # scaling factor for font size of text in graphs (among other things)
    "scaling": 1,
    # choice of colormap
    "cmap": "RdYlBu_r",
    # select image file format
    "format": "unset",
    # select image resolution
    "dpi": 200,
    # create binary file
    "binary": False,
}
