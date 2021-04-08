# nonos

[under construction]

### Installation

with python>=3.8.

````bash
git clone https://github.com/volodia99/nonos.git
cd nonos
python -m pip install .
````

A tool to analyze results from idefix/pluto/fargo3d simulations (for protoplanetary disks more specifically in 2D and 3D ONLY in cylindrical coordinates for now).

three possibilities : 
* use the command line tool
* use the local mode with the config.toml file
* create a script with functions/classes that are provided

TODO: write readme for the command line tool mode (ex: nonos -diff -on 10) / local mode (nonos -l + config.toml file)

Not implemented yet : 
* spherical coordinates
* error: `streamlines` & `midplane=False` -> not yet implemented
* warning: `isPlanet=False` & `corotate=True` -> we don't rotate the grid if there is no planet for now. `omegagrid` = 0.
* warning: `cartesian=False` & `midplane=False` -> plot not optimized for now in the (R,z) plane in polar.

### 1. Use of the command line tool

The default parameters are provided in config.toml file. Don't change this file directly.
````bash
nonos -mod d/f [options]
````
`-info`: give the default parameters in the config.toml file.  
`-dir`: where .vtk files and the inifile are stored (`"."` by default).  
`-mod [d/f]`: display/film (`""` home page by default).  
`-f [str]`: field (for now `RHO`, `VX1` and `VX2` in 2D, + `VX3` in 3D, `RHO` by default).  
`-on [int]`: if `-mod d` -> we plot the field of the data.on.vtk file (`1` by default).  
`-onend [int]`: if `-mod f` and `-partial` (`15` by default).  
`-partial`: if `-mod f` -> partial movie between `-on` and `-onend` (`false` by default).  
`-vmin [float]`: minimum value for the data (`-0.5` by default).  
`-vmax [float]`: maximum value for the data (`0.5` by default).  
`-diff`: plot the relative perturbation of the field f, i.e. `(f-f0)/f0` (`false` by default).  
`-log`: plot the log of the field f, i.e. `log(f)` (`false` by default).  
`-cor`: does the grid corotate? For now, works in pair with `-isp` (`false` by default).  
`-isp`: is there a planet in the grid ? (`false` by default)  
`-p [1d/2d]`: 1D axisymmetric radial profile or 2D field (`2d` by default).  
`-mid`/`-rz`: 2D plot in the (R-phi) plane or in the (R-z) plane (`-mid` by default).  
`-cart`/`-pol`: 2D plot in cartesian or polar coordinates (`-cart` by default).  
`-avr`/`-noavr`: do we average is the 3rd dimension, i.e. vertically when `-mid` and azimuthally when `-rz` (`-avr` by default).  
`-s`: do we compute streamlines? (`false` by default)  
`-stype [random/fixed/lic]`: do we compute random, fixed streams, or do we use line integral convolution? (`random` by default)  
`-srmin [float]`: minimum radius for streamlines computation (`0.7` by default).  
`-srmax [float]`: maximum radius for streamlines computation (`1.3` by default).  
`-sn [int]`: number of streamlines (`50` by default).  
`-ft [float]`: fontsize in the graph (`11` by default).  
`-cmap [str]`: choice of colormap for the `-p 2d` maps (`RdYlBu_r` by default).  
`-pbar`: do we display the progress bar when `-mod f`? (`false` by default)  
`-multi`: load and save figures in parallel when `-mod f` (`false` by default).  
`-cpu [int]`: number of cpus if `-multi` (`4` by default).  
`-l`: local mode.  

### 2. Use of the local mode

````bash
nonos -l
````
will copy locally the config.toml file. You can then open it directly (e.g., `atom config.toml`), and change the parameters.
Then run again:
````bash
nonos -l
````

### 3. Use of functions & classes

#### Example of field
````python
from nonos import InitParamNonos, FieldNonos
init = InitParamNonos(info=True) #info=True gives the default parameters in the param file config.toml
fieldon = FieldNonos(init, field='RHO', on=25, diff=True)
````
#### Example of plot without streamlines
````python
from nonos import InitParamNonos, PlotNonos
import matplotlib.pyplot as plt

init = InitParamNonos()
ploton = PlotNonos(init, field='RHO', on=25, diff=True)

fig, ax = plt.subplots()
ploton.plot(ax, cartesian=True, fontsize=8)
plt.show()
````
#### Example of (R,z) plot with quiver
````python
from nonos import InitParamNonos, FieldNonos, PlotNonos, StreamNonos
import matplotlib.pyplot as plt
import numpy as np

init = InitParamNonos(isPlanet=True, corotate=True)

ploton = PlotNonos(init, field='RHO', on=25, diff=True)
streamon = StreamNonos(init, on=25)
vx1on = FieldNonos(init, field='VX1', on=25, diff=False, log=False)
vr = vx1on.data
vx2on = FieldNonos(init, field='VX3', on=25, diff=False, log=False)
vz = vx2on.data

fig, ax = plt.subplots()
ploton.plot(ax, vmin=-0.15, vmax=0.15, midplane=False, cartesian=True, fontsize=8)
Z,R = np.meshgrid(ploton.zmed,ploton.xmed)
ax.quiver(R[:,::2],Z[:,::2],vr[:,ploton.ny//2,::2],vz[:,ploton.ny//2,::2])
plt.show()
````
#### Example of plot with streamlines with a planet
````python
from nonos import InitParamNonos, FieldNonos, PlotNonos, StreamNonos
import matplotlib.pyplot as plt

init = InitParamNonos(isPlanet=True, corotate=True)

ploton = PlotNonos(init, field='RHO', on=25, diff=True)
streamon = StreamNonos(init, on=25)
vx1on = FieldNonos(init, field='VX1', on=25, diff=False, log=False)
vr = vx1on.data
vx2on = FieldNonos(init, field='VX2', on=25, diff=False, log=False)
vphi = vx2on.data
vphi -= vx2on.omegaplanet[25]*vx2on.xmed[:,None,None]
streams = streamon.get_random_streams(vr,vphi,xmin=0.7,xmax=1.3, n=30)

fig, ax = plt.subplots()
ploton.plot(ax, cartesian=True, fontsize=8)
streamon.plot_streams(ax,streams,cartesian=True, color='k', linewidth=2, alpha=0.5)
plt.show()
````
#### Example of plot with a comparison between several simulations
````python
from nonos import InitParamNonos, PlotNonos
import matplotlib.pyplot as plt

dirpath=['path_to_dir1', 'path_to_dir2', 'path_to_dir3']

fig, ax = plt.subplots(figsize=(9,2.5), ncols=len(dirpath))
plt.subplots_adjust(left=0.05, right=0.94, top=0.95, bottom=0.1, wspace=0.4)
for i in range(len(dirpath)):
    init = InitParamNonos(directory=dirpath[i])
    ploton = PlotNonos(init, field='RHO', on=10, diff=True, directory=dirpath[i])
    ploton.plot(ax[i], cartesian=True, fontsize=6)

plt.show()
````
