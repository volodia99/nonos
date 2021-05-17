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
    # default min for radial extent
    "rmin": "unset",
    # default max for radial extent
    "rmax": "unset",
    # default min for vertical extent
    "zmin": "unset",
    # default max for vertical extent
    "zmax": "unset",
    # default min field when diff
    "vmin": "unset",
    # default max field when diff
    "vmax": "unset",
    # plane to represent (default is midplane)
    "rz": False,
    # do we average in the 3rd dimension?
    "noaverage": False,
    # do we compute random streams ("random"), fixed streams ("fixed") or lic ("lic")?
    "streamtype": "unset",
    # min radius for streamlines computation
    "rminStream": 0.7,
    # max radius for streamlines computation
    "rmaxStream": 1.3,
    # number of streamlines to draw
    "nstreamlines": 50,
    # do we display the progress (loading+plotting)
    "progressBar": False,
    # do we display the grid?
    "grid": False,
    # cartesian or polar coordinates
    "geometry": "cartesian",
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
}
