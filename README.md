# nonos
A tool to analyze results from idefix/pluto/fargo3d simulations (for protoplanetary disks more specifically)

#### Example of field
````python
from nonos import AnalysisNonos, FieldNonos
pconfig = AnalysisNonos(info=True).config #info=True gives the default parameters in the param file for pconfig
fieldon = FieldNonos(pconfig, field='RHO', on=25, diff=True)
````
#### Example of plot without streamlines
````python
from nonos import AnalysisNonos, PlotNonos
import matplotlib.pyplot as plt

pconfig = AnalysisNonos().config
ploton = PlotNonos(pconfig, field='RHO', on=25, diff=True)

fig, ax = plt.subplots()
ploton.plot(ax, cartesian=True, fontsize=8)
plt.show()
````
#### Example of plot with streamlines with a planet
````python
from nonos import AnalysisNonos, FieldNonos, PlotNonos, StreamNonos
import matplotlib.pyplot as plt

pconfig = AnalysisNonos().config
pconfig['isPlanet']=True
pconfig['corotate']=True

ploton = PlotNonos(pconfig, field='RHO', on=25, diff=True)
streamon = StreamNonos(pconfig, on=25)
vx1on = FieldNonos(pconfig, field='VX1', on=25, diff=False, log=False)
vr = vx1on.data
vx2on = FieldNonos(pconfig, field='VX2', on=25, diff=False, log=False)
vphi = vx2on.data
vphi -= vx2on.omegaplanet*vx2on.xmed[:,None,None]
streams = streamon.get_random_streams(vr,vphi,xmin=0.7,xmax=1.3, n=30)

fig, ax = plt.subplots()
ploton.plot(ax, cartesian=True, fontsize=8)
streamon.plot_streams(ax,streams,cartesian=True, color='k', linewidth=2, alpha=0.5)
plt.show()
````
#### Example of plot with a comparison between several simulations
````python
from nonos import AnalysisNonos, PlotNonos
import matplotlib.pyplot as plt

dirpath=['path_to_dir1', 'path_to_dir2', 'path_to_dir3']

fig, ax = plt.subplots(figsize=(8,6), ncols=len(dirpath))
plt.subplots_adjust(left=0.05, right=0.94, top=0.95, bottom=0.1, wspace=0.4)
for i in range(len(dirpath)):
    pconfig = AnalysisNonos(directory=dirpath[i]).config
    ploton = PlotNonos(pconfig, field='RHO', on=10, diff=True, directory=dirpath[i])
    ploton.plot(ax[i], cartesian=True, fontsize=6)

plt.show()
````
