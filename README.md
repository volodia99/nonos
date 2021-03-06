# nonos
[![PyPI](https://img.shields.io/pypi/v/nonos)](https://pypi.org/project/nonos/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/volodia99/nonos/main.svg)](https://results.pre-commit.ci/badge/github/volodia99/nonos/main.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

nonos is a 2D visualization command line application for planet-disk hydro simulations, as well as a Python library.
It works seamlessly with vtu-formatted data from Pluto, Fargo3D and Idefix.

:construction: This project and documentation are under construction :construction:
## Ongoing progress

* spherical coordinates
* error: `streamlines` & `rz=True` -> not yet implemented
* warning: `isPlanet=False` & `corotate=True` -> we don't rotate the grid if there is no planet for now. `omegagrid` = 0.
* warning: `geometry="polar"` & `rz=True` -> plot not optimized for now in the (R,z) plane in polar.


## Installation

:warning: Nonos requires Python 3.8 or newer. The easiest installation method is

```bash
$ pip install nonos
```

## Usage
### 1. On the command line

The nonos CLI gets its parameters from three sources:
- command line parameters
- a configuration file
- default values

Command line parameters take priority over the configuration file, which itself takes priority over default values.

To get help, run
```shell
$ nonos --help
```
```
usage: nonos [-h] [-dir DATADIR] [-field {RHO,VX1,VX2,VX3}] [-vmin VMIN]
             [-vmax VMAX] [-cpu NCPU] [-on ON [ON ...] | -all] [-diff] [-log]
             [-isp] [-corotate] [-grid] [-streamlines] [-rz] [-noavr] [-pbar]
             [-stype {random,fixed,lic}] [-srmin RMINSTREAM]
             [-srmax RMAXSTREAM] [-sn NSTREAMLINES]
             [-geom {cartesian,polar} | -pol] [-dim {1,2}] [-ft FONTSIZE]
             [-cmap CMAP] [-fmt FORMAT] [-dpi DPI] [-input INPUT | -isolated]
             [-d | -version | -logo | -config]

Analysis tool for idefix/pluto/fargo3d simulations (in polar coordinates).

optional arguments:
  -h, --help            show this help message and exit
  -dir DATADIR          location of output files and param files (default:
                        '.').
  -field {RHO,VX1,VX2,VX3}
                        name of field to plot (default: 'RHO').
  -vmin VMIN            min value in -diff mode (default: -0.5)
  -vmax VMAX            max value in -diff mode (default: -0.5)
  -cpu NCPU, -ncpu NCPU
                        number of parallel processes (default: 1).
  -on ON [ON ...]       output number(s) (on) to plot. This can be a single
                        value or a range (start, end, [step]) where both ends
                        are inclusive. (default: last output available).
  -all                  save an image for every available snapshot (this will
                        force show=False).
  -geom {cartesian,polar}
  -pol                  shortcut for -geom=polar
  -dim {1,2}            dimensionality in projection: 1 for a line plot, 2
                        (default) for a map.
  -ft FONTSIZE          fontsize in the graph (default: 11).
  -cmap CMAP            choice of colormap for the -dim 2 maps (default:
                        'RdYlBu_r').
  -fmt FORMAT, -format FORMAT
                        select output image file format (default: unset)
  -dpi DPI              image file resolution (default: DEFAULTS['dpi'])

boolean flags:
  -diff                 plot the relative perturbation of the field f, i.e.
                        (f-f0)/f0.
  -log                  plot the log10 of the field f, i.e. log(f).
  -isp                  is there a planet in the grid ?
  -corotate             does the grid corotate? Works in pair with -isp.
  -grid                 show the computational grid.
  -streamlines          plot streamlines.
  -rz                   2D plot in the (R-z) plane (default: represent the
                        midplane).
  -noavr, -noaverage    do not perform averaging along the third dimension.
  -pbar                 display a progress bar

streamlines options:
  -stype {random,fixed,lic}, -streamtype {random,fixed,lic}
                        streamlines method (default: 'unset')
  -srmin RMINSTREAM     minimum radius for streamlines computation (default:
                        0.7).
  -srmax RMAXSTREAM     maximum radius for streamlines computation (default:
                        1.3).
  -sn NSTREAMLINES      number of streamlines (default: 50).

CLI-only options:
  -input INPUT, -i INPUT
                        specify a configuration file.
  -isolated             ignore any existing 'nonos.toml' file.
  -d, -display          open a graphic window with the plot (only works with a
                        single image)
  -version, --version   show raw version number and exit
  -logo                 show Nonos logo with version number, and exit.
  -config               show configuration and exit.
```

#### Using a configuration file

The CLI will read parameters from a local file named `nonos.toml` if it exists,
or any other name specified using the `-i/-input` parameter.
To ignore any existing `nonos.toml` file, use the `-isolated` flag.

One way to configure nonos is to use
```shell
$ nonos -config
```

which prints the current configuration to stdout.
You can then redirect it to get a working configuration file as
```shell
$ nonos -config > nonos.toml
```
This method can also be used to store a complete configuration file from command line arguments:
```shell
$ nonos -ncpu 8 -cmap viridis -rz -diff -vmin=-10 -vmax=+100 -config
```
As of Nonos 0.2.0, this will print
```
# Generated with nonos 0.2.0
datadir               =  "."
field                 =  "RHO"
dimensionality        =  2
on                    =  "unset"
diff                  =  true
log                   =  false
vmin                  =  -10.0
vmax                  =  100.0
rz                    =  true
noaverage             =  false
streamtype            =  "unset"
rminStream            =  0.7
rmaxStream            =  1.3
nstreamlines          =  50
progressBar           =  false
grid                  =  false
geometry              =  "cartesian"
isPlanet              =  false
corotate              =  false
ncpu                  =  8
fontsize              =  11
cmap                  =  "viridis"
dpi                   =  200
````

### 2. Programmatic usage

We are still working on nonos' api.
