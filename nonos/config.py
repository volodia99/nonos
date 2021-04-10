DEFAULTS = {
    # directory where the simulation results are stored
    'datadir': '.',
    # mode can be d (display), f (film)
    'mode': 'd',
    # which field?
    'field': 'RHO',
    # 1D axisymmetric porfile or 2D map
    'dimensionality': 2,
    # first output number ('on') to plot
    'onStart': 1,
    # last output number to use in film mode
    'onEnd': 'unset',
    # output number interval between used snapshot in film mode
    'onStep': 1,
    # perturbation of field
    'diff': False,
    # field in log
    'log': False,
    # default min field when diff
    'vmin': -0.5,
    # default max field when diff
    'vmax': 0.5,
    # plane to represent (default is midplane)
    'rz': False,
    # do we average in the 3rd dimension?
    'noaverage': False,
    # do we compute random streams ("random"), fixed streams ("fixed") or lic ("lic")?
    'streamtype': 'unset',
    # min radius for streamlines computation
    'rminStream': 0.7,
    # max radius for streamlines computation
    'rmaxStream': 1.3,
    # number of streamlines to draw
    'nstreamlines': 50,
    # do we display the progress (loading+plotting)
    'progressBar': False,
    # do we display the grid?
    'grid': False,
    # cartesian or polar coordinates
    'geometry': "cartesian",
    # is there a planet in the grid?
    'isPlanet': False,
    # do the grid rotate with the planet?
    'corotate': False,
    # number of cpus to use
    'ncpu': 1,
    # font size of text in graphs
    'fontsize': 11,
    # choice of colormap
    'cmap': 'RdYlBu_r',
}
