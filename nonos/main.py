#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adapted from pbenitez-llambay, gwafflard-fernandez, crobert-lot & glesur
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool, Value

import toml
import inifix as ix
import functools
from rich import print as rprint
import lic
import glob
from typing import List, Optional
import argparse

# TODO: check in 3D
# TODO: adapt config['midplane'] as an arg when 3D
# TODO: check in plot function if corotate=True works for all vtk and dpl (initial planet location) -> computation to calculate the grid rotation speed
# TODO: compute gas surface density and not just gas volume density : something like self.data*=np.sqrt(2*np.pi)*self.aspectratio*self.y
# TODO: check how to generalize the path of the directory (.toml file)
# TODO: compute vortensity
# TODO: compute vertical flows (cf vertical_flows.txt)
# TODO: adapt streamline analysis (cf strl.py)
# TODO: re-check if each condition works fine
# TODO: CORRECTED AND TO BE TESTED maybe small bug in the progress bar when nbcpu=n_files
# TODO: recheck the axiplot feature (parallelization, ...) and the writeField feature
# TODO: streamlines does not work properly (azimuthal displacement due to reconstruction of field with corotation)
# TODO: streamline analysis: test if the estimation of the radial spacing works
# TODO: major modif for the corotation implementation
# TODO: streamlines = 'random', 'specific' or 'lic'
# TODO: write a better way to save pictures (function in PlotNonos maybe)
# TODO: dor now, calculations are made only in working directory, need to specify directory in command lines
# TODO: do not forget to change all the functions that use dpl (planet location), which is valid if the planet is in a fixed cicular orbit

firstRun = True

class DataStructure:
    pass

def readVTKPolar(filename, cell='edges'):
    """
    Adapted from Geoffroy Lesur
    Function that reads a vtk file in polar coordinates
    """
    try:
        fid=open(filename,"rb")
    except:
        print_err("Can't open vtk file")
        return 1

    # define our datastructure
    V=DataStructure()

    # raw data which will be read from the file
    V.data={}

    # datatype we read
    dt=np.dtype(">f")   # Big endian single precision floats

    s=fid.readline()    # VTK DataFile Version x.x
    s=fid.readline()    # Comments

    s=fid.readline()    # BINARY
    s=fid.readline()    # DATASET RECTILINEAR_GRID

    slist=s.split()
    grid_type=str(slist[1],'utf-8')
    if(grid_type != "STRUCTURED_GRID"):
        fid.close()
        print_err("Wrong VTK file type.\nCurrent type is: %s.\nThis routine can only open Polar VTK files."%(grid_type))
        return 1

    s=fid.readline()    # DIMENSIONS NX NY NZ
    slist=s.split()
    V.nx=int(slist[1])
    V.ny=int(slist[2])
    V.nz=int(slist[3])
    # print("nx=%d, ny=%d, nz=%d"%(V.nx,V.ny,V.nz))

    s=fid.readline()    # POINTS NXNYNZ float
    slist=s.split()
    npoints=int(slist[1])
    points=np.fromfile(fid,dt,3*npoints)
    s=fid.readline()    # EXTRA LINE FEED

    V.points=points

    if V.nx*V.ny*V.nz != npoints:
        print_err("Grid size incompatible with number of points in the data set")
        return 1

    # Reconstruct the polar coordinate system
    x1d=points[::3]
    y1d=points[1::3]
    z1d=points[2::3]

    xcart=np.transpose(x1d.reshape(V.nz,V.ny,V.nx))
    ycart=np.transpose(y1d.reshape(V.nz,V.ny,V.nx))
    zcart=np.transpose(z1d.reshape(V.nz,V.ny,V.nx))

    r=np.sqrt(xcart[:,0,0]**2+ycart[:,0,0]**2)
    theta=np.unwrap(np.arctan2(ycart[0,:,0],xcart[0,:,0]))
    z=zcart[0,0,:]

    s=fid.readline()    # CELL_DATA (NX-1)(NY-1)(NZ-1)
    slist=s.split()
    data_type=str(slist[0],'utf-8')
    if(data_type != "CELL_DATA"):
        fid.close()
        print_err("this routine expect CELL DATA as produced by PLUTO.")
        return 1
    s=fid.readline()    # Line feed

    if cell=='edges':
        if V.nx>1:
            V.nx=V.nx-1
            V.x=r
        else:
            V.x=r
        if V.ny>1:
            V.ny=V.ny-1
            V.y=theta
        else:
            V.y=theta
        if V.nz>1:
            V.nz=V.nz-1
            V.z=z
        else:
            V.z=z

    # Perform averaging on coordinate system to get cell centers
    # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
    elif cell=='centers':
        if V.nx>1:
            V.nx=V.nx-1
            V.x=0.5*(r[1:]+r[:-1])
        else:
            V.x=r
        if V.ny>1:
            V.ny=V.ny-1
            V.y=(0.5*(theta[1:]+theta[:-1])+np.pi)%(2.0*np.pi)-np.pi
        else:
            V.y=theta
        if V.nz>1:
            V.nz=V.nz-1
            V.z=0.5*(z[1:]+z[:-1])
        else:
            V.z=z

    while 1:
        s=fid.readline()        # SCALARS/VECTORS name data_type (ex: SCALARS imagedata unsigned_char)
        #print repr(s)
        if len(s)<2:         # leave if end of file
            break
        slist=s.split()
        datatype=str(slist[0],'utf-8')
        varname=str(slist[1],'utf-8')
        if datatype == "SCALARS":
            fid.readline()  # LOOKUP TABLE
            V.data[varname] = np.transpose(np.fromfile(fid,dt,V.nx*V.ny*V.nz).reshape(V.nz,V.ny,V.nx))
        elif datatype == "VECTORS":
            Q=np.fromfile(fid,dt,3*V.nx*V.ny*V.nz)

            V.data[varname+'_X']=np.transpose(Q[::3].reshape(V.nz,V.ny,V.nx))
            V.data[varname+'_Y']=np.transpose(Q[1::3].reshape(V.nz,V.ny,V.nx))
            V.data[varname+'_Z']=np.transpose(Q[2::3].reshape(V.nz,V.ny,V.nx))

        else:
            print_err("Unknown datatype %s" % datatype)
            return 1
            break;

        fid.readline()  #extra line feed
    fid.close()

    return V

class Parameters():
    """
    Adapted from Pablo Benitez-Llambay
    Class for reading the simulation parameters.
    input: string -> name of the parfile, normally *.ini
    """
    def __init__(self, config, directory="", paramfile=None, corotate=None):
        if corotate is None:
            corotate=self.config['corotate']
        if paramfile is None:
            try:
                paramfile = "idefix.ini"
                params = open(os.path.join(directory,paramfile),'r') #Opening the parfile
                self.code = 'idefix'
            except IOError:                  # Error checker.
                try:
                    paramfile = "pluto.ini"
                    params = open(os.path.join(directory,paramfile),'r') #Opening the parfile
                    self.code = 'pluto'
                except IOError:                  # Error checker.
                    try:
                        paramfile = "variables.par"
                        params = open(os.path.join(directory,paramfile),'r') #Opening the parfile
                        self.code = 'fargo3d'
                    except IOError:                  # Error checker.
                        print_err("idefix.ini, pluto.ini or variables.par not found.")
                        return 1
        else:
            print_err("For now, impossible to choose your parameter file.\nBy default, the code searches idefix.ini then pluto.ini then variables.par.")
            return 1

        self.paramfile = paramfile
        self.iniconfig = ix.load(os.path.join(directory,self.paramfile))

        if self.code=='idefix':
            self.vtk = self.iniconfig["Output"]["vtk"]
            if config['isPlanet']:
                self.qpl = self.iniconfig["Planet"]["qpl"]
                self.dpl = self.iniconfig["Planet"]["dpl"]
                self.omegaplanet = np.sqrt((1.0+self.qpl)/self.dpl/self.dpl/self.dpl)
                if corotate:
                    self.omegagrid = self.omegaplanet
            else:
                if corotate:
                    self.omegagrid = 0.0

        elif self.code=='pluto':
            self.vtk = self.iniconfig["Static Grid Output"]["vtk"][0]
            if config['isPlanet']:
                self.qpl = self.iniconfig["Parameters"]["Mplanet"]/self.iniconfig["Parameters"]["Mstar"]
                print_warn("Initial distance not defined in pluto.ini.\nBy default, dpl=1.0 for the computation of omegaP\n")
                self.dpl = 1.0
                self.omegaplanet = np.sqrt((1.0+self.qpl)/self.dpl/self.dpl/self.dpl)
                if corotate:
                    self.omegagrid = self.omegaplanet
            else:
                if corotate:
                    self.omegagrid = 0.0

        elif self.code=='fargo3d':
            try:
                cfgfile = glob.glob1(directory,"*.cfg")[0]
            except IndexError:
                print_err("*.cfg file (FARGO3D planet parameters) does not exist in %s directory"%directory)
                return 1

            self.cfgconfig = ix.load(os.path.join(directory,cfgfile))
            self.vtk = self.iniconfig["NINTERM"]*self.iniconfig["DT"]
            if config['isPlanet']:
                self.qpl = self.cfgconfig[list(self.cfgconfig)[0]][1]
                self.dpl = self.cfgconfig[list(self.cfgconfig)[0]][0]
                self.omegaplanet = np.sqrt((1.0+self.qpl)/self.dpl/self.dpl/self.dpl)
                if corotate:
                    self.omegagrid = self.omegaplanet
            else:
                if corotate:
                    self.omegagrid = 0.0

class AnalysisNonos(Parameters):
    """
    read the .toml file
    find parameters in config.toml (same directory as script)
    compute the number of data.*.vtk files in working directory
    """
    def __init__(self, directory_of_script=os.path.dirname(os.path.abspath(__file__)), directory="", info=False, paramfile=None):
        self.directory = directory
        try:
            self.config=toml.load(os.path.join(directory_of_script,"config.toml"))
            if info:
                print('\n')
                print('--------------------------------------')
                for keys in self.config:
                    print("config['%s'] = %s"%(keys,self.config[keys]))
                    # for subkeys in config[keys]:
                    #     print("config['%s']['%s'] = %s"%(keys,subkeys,config[keys][subkeys]))
                print('--------------------------------------')
                print('\n')
        except FileNotFoundError:
            print_err(os.path.join(directory_of_script,"config.toml")+" not found")
            return 1

        if(not(self.config['midplane']) and self.config['corotate']):
            print_err("corotate is not yet implemented in the (R,z) plane")
            return 1

        if(not(self.config['average']) and not(self.config['midplane'])):
            print_err("average=False is not yet implemented in the (R,z) plane")
            return 1

        if(not(self.config['isPlanet']) and self.config['corotate']):
            print_warn("We don't rotate the grid if there is no planet for now.\nomegagrid = 0.")

        if(self.config['streamlines'] and self.config['streamtype']=='lic'):
            print_warn("TODO: check what is the length argument in StreamNonos().get_lic_streams ?")

        Parameters.__init__(self, config=self.config, directory=self.directory, paramfile=paramfile) #All the Parameters attributes inside Field
        if info:
            print("\n")
            print(self.code.upper(), "analysis")

        if (self.code=='idefix' or self.code=='pluto'):
            if info:
                print("\nWORKS IN POLAR COORDINATES")
            try:
                domain=readVTKPolar(os.path.join(self.directory,'data.0000.vtk'), cell="edges")
                list_keys=list(domain.data.keys())
                if info:
                    print("Possible fields: ", list_keys)
                    print('nR=%d, np=%d, nz=%d' % (domain.nx,domain.ny,domain.nz))
            except IOError:
                print_err("IOError with data.0000.vtk")
                return 1

            self.n_file = len(glob.glob1(self.directory,"data.*.vtk"))

        elif(self.code=='fargo3d'):
            if info:
                print("\nWORKS IN POLAR COORDINATES")
            try:
                domain_x = np.loadtxt(os.path.join(self.directory,"domain_x.dat"))
            except IOError:
                print_err("IOError with domain_x.dat")
                return 1
            try:
                #We avoid ghost cells
                domain_y = np.loadtxt(os.path.join(self.directory,"domain_y.dat"))[3:-3]
            except IOError:
                print_err("IOError with domain_y.dat")
                return 1
            try:
                domain_z = np.loadtxt(os.path.join(self.directory,"domain_z.dat"))
                if domain_z.shape[0]>6:
                    domain_z=domain_z[3:-3]
            except IOError:
                print_err("IOError with domain_z.dat")
                return 1

            self.n_file = len(glob.glob1(self.directory,"gasdens*.dat")) - len(glob.glob1(self.directory,"gasdens*_*.dat"))

class Mesh(Parameters):
    """
    Adapted from Pablo Benitez-Llambay
    Mesh class, for keeping all the mesh data.
    Input: directory [string] -> this is where the domain files are.
    """
    def __init__(self, config, directory="", paramfile=None):
        Parameters.__init__(self, config=config, directory=directory, paramfile=paramfile) #All the Parameters attributes inside Field
        if (self.code=='idefix' or self.code=='pluto'):
            try:
                domain=readVTKPolar(os.path.join(directory,'data.0000.vtk'), cell="edges")
            except IOError:
                print_err("IOError with data.0000.vtk")
                return 1

            self.domain = domain

            self.nx = self.domain.nx
            self.ny = self.domain.ny
            self.nz = self.domain.nz

            self.xedge = self.domain.x #X-Edge
            self.yedge = self.domain.y-np.pi #Y-Edge
            self.zedge = self.domain.z #Z-Edge

            # index of the cell in the midplane
            self.imidplane = self.nz//2

        elif(self.code=='fargo3d'):
            try:
                domain_x = np.loadtxt(os.path.join(directory,"domain_x.dat"))
            except IOError:
                print_err("IOError with domain_x.dat")
                return 1
            try:
                #We avoid ghost cells
                domain_y = np.loadtxt(os.path.join(directory,"domain_y.dat"))[3:-3]
            except IOError:
                print_err("IOError with domain_y.dat")
                return 1
            try:
                domain_z = np.loadtxt(os.path.join(directory,"domain_z.dat"))
                if domain_z.shape[0]>6:
                    domain_z=domain_z[3:-3]
            except IOError:
                print_err("IOError with domain_z.dat")
                return 1

            self.xedge = domain_y #X-Edge
            self.yedge = domain_x #Y-Edge
            # self.zedge = np.pi/2-domain_z #Z-Edge #latitute
            self.zedge = domain_z #Z-Edge #latitute

            self.nx=len(self.xedge)-1
            self.ny=len(self.yedge)-1
            self.nz=len(self.zedge)-1

            if np.sign(self.zedge[0])!=np.sign(self.zedge[-1]):
                self.imidplane = self.nz//2
            else:
                self.imidplane = -1

        self.xmed = 0.5*(self.xedge[1:]+self.xedge[:-1]) #X-Center
        self.ymed = 0.5*(self.yedge[1:]+self.yedge[:-1]) #Y-Center
        self.zmed = 0.5*(self.zedge[1:]+self.zedge[:-1]) #Z-Center

        # width of each cell in all directions
        self.dx = np.ediff1d(self.xedge)
        self.dy = np.ediff1d(self.yedge)
        self.dz = np.ediff1d(self.zedge)

        self.x = self.xedge
        self.y = self.yedge
        self.z = self.zedge

class FieldNonos(Mesh,Parameters):
    """
    Inspired by Pablo Benitez-Llambay
    Field class, it stores the mesh, parameters and scalar data
    for a scalar field.
    Input: field [string] -> filename of the field
           directory='' [string] -> where filename is
    """
    def __init__(self, config, directory="", field=None, on=None, paramfile=None, diff=None, log=None, corotate=None):
        self.config = config
        Mesh.__init__(self, config=self.config, directory=directory, paramfile=paramfile)       #All the Mesh attributes inside Field
        Parameters.__init__(self, config=self.config, directory=directory, paramfile=paramfile, corotate=corotate) #All the Parameters attributes inside Field

        if field is None:
            field=self.config['field']
        if on is None:
            on=self.config['onStart']
        if diff is None:
            diff=self.config['diff']
        if log is None:
            log=self.config['log']
        if corotate is None:
            corotate=self.config['corotate']

        self.on = on
        self.diff=diff
        self.log=log
        self.corotate=corotate

        self.field = field
        filedata = "data.%04d.vtk"%self.on
        filedata0 = "data.0000.vtk"
        if self.code=='pluto':
            self.field=self.field.lower()
        elif self.code=='fargo3d':
            if self.field=='RHO':
                self.field='dens'
            if self.field=='VX1':
                self.field='vy'
            if self.field=='VX2':
                self.field='vx'
            if self.field=='VX3':
                self.field='vz'
            filedata = "gas%s%d.dat"%(self.field,self.on)
            filedata0 = "gas%s0.dat"%self.field

        self.data = self.__open_field(os.path.join(directory,filedata)) #The scalar data is here.
        self.data0 = self.__open_field(os.path.join(directory,filedata0))

        if self.log:
            if self.diff:
                self.data = np.log10(self.data/self.data0)
                self.title = r'log($\frac{%s}{%s_0}$)'%(self.field,self.field)
            else:
                self.data = np.log10(self.data)
                self.title = 'log(%s)'%self.field
        else:
            if self.diff:
                self.data = (self.data-self.data0)/self.data0
                self.title = r'$\frac{%s - %s_0}{%s_0}$'%(self.field,self.field,self.field)
            else:
                self.data = self.data
                self.title = '%s'%self.field

    def __open_field(self, f):
        """
        Reading the data
        """
        if(self.code=='idefix' or self.code=='pluto'):
            data = readVTKPolar(f, cell='edges').data[self.field]
            data = np.concatenate((data[:,self.ny//2:self.ny,:], data[:,0:self.ny//2,:]), axis=1)
        elif(self.code=='fargo3d'):
            data = np.fromfile(f, dtype='float64')
            data=(data.reshape(self.nz,self.nx,self.ny)).transpose(1,2,0) #rad, pĥi, theta

        """
        if we try to rotate a grid at 0 speed
        and if the domain is exactly [-pi,pi],
        impossible to perform the following calculation (try/except)
        we therefore don't move the grid if the rotation speed is null
        """
        if (self.corotate and self.on*self.vtk*self.omegagrid!=0.0):
            P,R = np.meshgrid(self.y,self.x)
            Prot=P-(self.on*self.vtk*self.omegagrid)%(2*np.pi)
            try:
                index=(np.where(Prot[0]>np.pi))[0].min()
            except ValueError:
                index=(np.where(Prot[0]<-np.pi))[0].max()
            data=np.concatenate((data[:,index:self.ny,:],data[:,0:index,:]),axis=1)
        return (data)

class PlotNonos(FieldNonos):
    """
    Plot class which uses Field to compute different graphs.
    """
    def __init__(self, config, directory="", field=None, on=None, diff=None, log=None, corotate=None):
        FieldNonos.__init__(self,config=config,field=field,on=on,directory=directory, diff=diff, log=log, corotate=corotate) #All the Parameters attributes inside Field

    def axiplot(self, ax, vmin=None, vmax=None, fontsize=None, **karg):
        dataProfile=np.mean(np.mean(self.data,axis=1),axis=1)
        if vmin is None:
            vmin=self.config['vmin']
            if not(self.diff):
                vmin=dataProfile.min()
        if vmax is None:
            vmax=self.config['vmax']
            if not(self.diff):
                vmax=dataProfile.max()
        if fontsize is None:
            fontsize=self.config['fontsize']

        if self.config['writeAxi']:
            axifile=open("axi%s%04d.csv"%(self.field.lower(),self.on),'w')
            for i in range(len(self.xmed)):
                axifile.write('%f,%f\n' %(self.xmed[i],dataProfile[i]))
            axifile.close()

        ax.plot(self.xmed,dataProfile,**karg)

        if not(self.log):
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_ylim(vmin,vmax)
        ax.tick_params('both', labelsize=fontsize)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.set_xlabel('Radius', fontsize=fontsize)
        ax.set_ylabel(self.title, fontsize=fontsize)
        # plt.legend(frameon=False)

    def plot(self, ax, vmin=None, vmax=None, fontsize=None, cartesian=None, cmap=None, **karg):
        """
        A layer for pcolormesh function.
        """
        if vmin is None:
            vmin=self.config['vmin']
            if not(self.diff):
                vmin=self.data.min()
        if vmax is None:
            vmax=self.config['vmax']
            if not(self.diff):
                vmax=self.data.max()
        if cartesian is None:
            cartesian=self.config['cartesian']
        if fontsize is None:
            fontsize=self.config['fontsize']
        if cmap is None:
            cmap=self.config['cmap']

        # (R,phi) plane
        if self.config['midplane']:
            if cartesian:
                P,R = np.meshgrid(self.y,self.x)
                X = R*np.cos(P)
                Y = R*np.sin(P)
                if self.config['average']:
                    im=ax.pcolormesh(X,Y,np.mean(self.data,axis=2),
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)
                else:
                    im=ax.pcolormesh(X,Y,self.data[:,:,self.imidplane],
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)

                ax.set_aspect('equal')
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)
                ax.set_ylabel('Y [c.u.]', family='monospace', fontsize=fontsize)
                ax.set_xlabel('X [c.u.]', family='monospace', fontsize=fontsize)
                if self.config['grid']:
                    ax.plot(X,Y,c='k',linewidth=0.07)
                    ax.plot(X.transpose(),Y.transpose(),c='k',linewidth=0.07)
            else:
                P,R = np.meshgrid(self.y,self.x)
                if self.config['average']:
                    im=ax.pcolormesh(R,P,np.mean(self.data,axis=2),
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)
                else:
                    im=ax.pcolormesh(R,P,self.data[:,:,self.imidplane],
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)

                ax.set_ylim(-np.pi,np.pi)
                ax.set_aspect('auto')
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)
                ax.set_ylabel('Phi', family='monospace', fontsize=fontsize)
                ax.set_xlabel('Radius', family='monospace', fontsize=fontsize)
                if self.config['grid']:
                    ax.plot(R,P,c='k',linewidth=0.07)
                    ax.plot(R.transpose(),P.transpose(),c='k',linewidth=0.07)

        # (R,z) plane
        else:
            if cartesian:
                Z,R = np.meshgrid(self.z,self.x)
                if ['average']:
                    im=ax.pcolormesh(R,Z,np.mean(self.data,axis=1),
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)
                # else:
                #     print_warn("average=False is not yet implemented when midplane=False")
                #     sys.exit()
                    # im=ax.pcolormesh(R,Z,self.data[:,:,self.imidplane],
                    #           cmap=cmap,vmin=vmin,vmax=vmax,**karg)
                ax.set_aspect('auto')
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)
                ax.set_ylabel('Z [c.u.]', family='monospace', fontsize=fontsize)
                ax.set_xlabel('X [c.u.]', family='monospace', fontsize=fontsize)
                # ax.set_xlim(-6.0,6.0)
                # ax.set_ylim(-6.0,6.0)
                if self.config['grid']:
                    # im=ax.scatter(X,Y,c=np.mean(self.data,axis=2))
                    ax.plot(R,Z,c='k',linewidth=0.07)
                    ax.plot(R.transpose(),Z.transpose(),c='k',linewidth=0.07)
            else:
                Z,R = np.meshgrid(self.z,self.x)
                r = np.sqrt(R**2+Z**2)
                t = np.arctan2(R,Z)
                if self.config['average']:
                    im=ax.pcolormesh(r,t,np.mean(self.data,axis=1),
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)
                # else:
                #     print_warn("average=False is not yet implemented when midplane=False")
                #     sys.exit()
                    # im=ax.pcolormesh(r,t,self.data[:,:,self.imidplane],
                    #           cmap=cmap,vmin=vmin,vmax=vmax,**karg)

                ax.set_aspect('auto')
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)
                ax.set_ylabel('Theta', family='monospace', fontsize=fontsize)
                ax.set_xlabel('Radius', family='monospace', fontsize=fontsize)
                # ax.set_xlim(-6.0,6.0)
                # ax.set_ylim(-6.0,6.0)
                if self.config['grid']:
                    # im=ax.scatter(X,Y,c=np.mean(self.data,axis=2))
                    ax.plot(r,t,c='k',linewidth=0.07)
                    ax.plot(r.transpose(),t.transpose(),c='k',linewidth=0.07)

        # ax.set_xlim(0.5,1.5)
        # ax.set_ylim(-0.8,0.8)
        ax.set_title(self.code, family='monospace', fontsize=fontsize)
        ax.tick_params('both', labelsize=fontsize)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(im, cax=cax, orientation='vertical')#, format='%.0e')
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label(self.title, family='monospace', fontsize=fontsize)

class StreamNonos(FieldNonos):
    """
    Adapted from Pablo Benitez-Llambay
    Class which uses Field to compute streamlines.
    """
    def __init__(self, config, directory="", field=None, on=None):
        FieldNonos.__init__(self,config=config,field=field,on=on,directory=directory) #All the Parameters attributes inside Field

        if field is None:
            field=self.config['field']
        if on is None:
            on=self.config['onStart']

    def bilinear(self,x,y,f,p):
        """
        Bilinear interpolation.
        Parameters
        ----------
        x = (x1,x2); y = (y1,y2)
        f = (f11,f12,f21,f22)
        p = (x,y)
        where x,y are the interpolated points and
        fij are the values of the function at the
        points (xi,yj).
        Output
        ------
        f(p): Float.
              The interpolated value of the function f(p) = f(x,y)
        """

        xp  = p[0]; yp   = p[1]; x1  = x[0]; x2  = x[1]
        y1  = y[0]; y2  = y[1];  f11 = f[0]; f12 = f[1]
        f21 = f[2]; f22 = f[3]
        t = (xp-x1)/(x2-x1);    u = (yp-y1)/(y2-y1)

        return (1.0-t)*(1.0-u)*f11 + t*(1.0-u)*f12 + t*u*f22 + u*(1-t)*f21

    def get_v(self, v, x, y):
        """
        For a real set of coordinates (x,y), returns the bilinear
        interpolated value of a Field class.
        """

        i = find_nearest(self.x,x)
        # i = int(np.log10(x/self.x.min())/np.log10(self.x.max()/self.x.min())*self.nx)
        # i = int((x-self.x.min())/(self.x.max()-self.x.min())*self.nx)
        j = int((y-self.y.min())/(self.y.max()-self.y.min())*self.ny)

        if i<0 or j<0 or i>v.shape[0]-2 or j>v.shape[1]-2:
            return None

        f11 = v[i,j,self.imidplane]
        f12 = v[i,j+1,self.imidplane]
        f21 = v[i+1,j,self.imidplane]
        f22 = v[i+1,j+1,self.imidplane]
        try:
            x1  = self.x[i]
            x2  = self.x[i+1]
            y1  = self.y[j]
            y2  = self.y[j+1]
            return self.bilinear((x1,x2),(y1,y2),(f11,f12,f21,f22),(x,y))
        except IndexError:
            return None

    def euler(self, vx, vy, x, y, reverse):
        """
        Euler integrator for computing the streamlines.
        Parameters:
        ----------

        x,y: Floats.
             Initial condition
        reverse: Boolean.
                 If reverse is true, the integration step is negative.

        Output
        ------

        (dx,dy): (float,float).
                 Are the azimutal and radial increments.
                 Only works for cylindrical coordinates.
        """
        sign = 1.0
        if reverse:
            sign = -1
        vr = self.get_v(vx,x,y)
        vt = self.get_v(vy,x,y)
        if vt == None or vr == None: #Avoiding problems...
            return None,None

        l = np.min((((self.x.max()-self.x.min())/self.nx),((self.y.max()-self.y.min())/self.ny)))
        h = 0.5*l/np.sqrt((vr**2+vt**2))

        return sign*h*np.array([vr,vt/x])

    def get_stream(self, vx, vy, x0, y0, nmax=1000000, maxlength=4*np.pi, bidirectional=True, reverse=False):
        """
        Function for computing a streamline.
        Parameters:
        -----------

        x0,y0: Floats.
              Initial position for the stream
        nmax: Integer.
              Maxium number of iterations for the stream.
        maxlength: Float
                   Maxium allowed length for a stream
        bidirectional=True
                      If it's True, the stream will be forward and backward computed.
        reverse=False
                The sign of the stream. You can change it mannualy for a single stream,
                but in practice, it's recommeneded to use this function without set reverse
                and setting bidirectional = True.

        Output:
        -------

        If bidirectional is False, the function returns a single array, containing the streamline:
        The format is:

                                          np.array([[x],[y]])

        If bidirectional is True, the function returns a tuple of two arrays, each one with the same
        format as bidirectional=False.
        The format in this case is:

                                (np.array([[x],[y]]),np.array([[x],[y]]))

        This format is a little bit more complicated, and the best way to manipulate it is with iterators.
        For example, if you want to plot the streams computed with bidirectional=True, you can do:

        stream = get_stream(x0,y0)
        ax.plot(stream[0][0],stream[0][1]) #Forward
        ax.plot(stream[1][0],stream[1][1]) #Backward

        """

        if bidirectional:
            s0 = self.get_stream(vx, vy, x0, y0, reverse=False, bidirectional=False, nmax=nmax,maxlength=maxlength)
            s1 = self.get_stream(vx, vy, x0, y0, reverse=True,  bidirectional=False, nmax=nmax,maxlength=maxlength)
            return (s0,s1)

        l = 0
        x = [x0]
        y = [y0]

        for i in range(nmax):
            ds = self.euler(vx, vy, x0, y0, reverse=reverse)
            if(ds[0] == None):
                # if(len(x)==1):
                #     print_warn("There was an error getting the stream, ds is NULL (see get_stream).")
                break
            l += np.sqrt(ds[0]**2+ds[1]**2)
            dx = ds[0]
            dy = ds[1]
            if(np.sqrt(dx**2+dy**2)<1e-13):
                print_warn("(get_stream): ds is very small, check if you're in a stagnation point.\nTry selecting another initial point.")
                break
            if (l > maxlength):
                # print("maxlength reached: ", l)
                break
            x0 += dx
            y0 += dy
            x.append(x0)
            y.append(y0)

        return np.array([x,y])

    def get_random_streams(self, vx, vy, xmin=None, xmax=None, ymin=None, ymax=None, n=30, nmax=100000):
        if xmin == None:
            xmin = self.x.min()
        if ymin == None:
            ymin = self.y.min()
        if xmax == None:
            xmax = self.x.max()
        if ymax == None:
            ymax = self.y.max()

        X = xmin + np.random.rand(n)*(xmax-xmin)
        # X = xmin*pow((xmax/xmin),np.random.rand(n))
        Y = ymin + np.random.rand(n)*(ymax-ymin)

        streams = []
        counter = 0
        for x,y in zip(X,Y):
            stream = self.get_stream(vx, vy, x, y, nmax=nmax, bidirectional=True)
            streams.append(stream)
            counter += 1
        return streams

    def get_fixed_streams(self, vx, vy, xmin=None, xmax=None, ymin=None, ymax=None, n=30, nmax=100000):
        if xmin == None:
            xmin = self.x.min()
        if ymin == None:
            ymin = self.y.min()
        if xmax == None:
            xmax = self.x.max()
        if ymax == None:
            ymax = self.y.max()

        X = xmin + np.linspace(0,1,n)*(xmax-xmin)
        # X = xmin*pow((xmax/xmin),np.random.rand(n))
        Y = ymin + np.linspace(0,1,n)*(ymax-ymin)

        streams = []
        counter = 0
        for x,y in zip(X,Y):
            stream = self.get_stream(vx, vy, x, y, nmax=nmax, bidirectional=True)
            streams.append(stream)
            counter += 1
        return streams

    def plot_streams(self, ax, streams, cartesian=False, **kargs):
        for stream in streams:
            for sub_stream in stream:
                # sub_stream[0]*=unit_code.length/unit.AU
                if cartesian:
                    ax.plot(sub_stream[0]*np.cos(sub_stream[1]),sub_stream[0]*np.sin(sub_stream[1]),**kargs)
                else:
                    ax.plot(sub_stream[0],sub_stream[1],**kargs)

    def get_lic_streams(self, vx, vy):
        get_lic=lic.lic(vx[:,:,self.imidplane],vy[:,:,self.imidplane],length=30)
        return(get_lic)

    def plot_lic(self, ax, streams, cartesian=False, **kargs):
        if cartesian:
            P,R = np.meshgrid(self.y,self.x)
            X = R*np.cos(P)
            Y = R*np.sin(P)
            ax.pcolormesh(X,Y,streams,**kargs)
        else:
            P,R = np.meshgrid(self.y,self.x)
            ax.pcolormesh(R,P,streams,**kargs)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def print_warn(message):
    """
    from crobert
    """
    rprint(f"[bold red]Warning |[/] {message}", file=sys.stderr)

def print_err(message):
    """
    from crobert
    """
    rprint(f"[bold white on red]Error |[/] {message}", file=sys.stderr)

# process function for parallisation purpose with progress bar
counter = Value('i', 0) # initialization of a counter
def process_field(on, profile, field, cart, diff, log, corotate, streamlines, stype, srmin, srmax, nstream, config, vmin, vmax, isPlanet, pbar, parallel, directory):
    ploton=PlotNonos(config, field=field, on=on, diff=diff, log=log, corotate=corotate, directory=directory)
    if streamlines:
        streamon=StreamNonos(config, field=field, on=on, directory=directory)
        vx1on = FieldNonos(config, field='VX1', on=on, diff=False, log=False, corotate=False, directory=directory)
        vx2on = FieldNonos(config, field='VX2', on=on, diff=False, log=False, corotate=False, directory=directory)
    fig, ax=plt.subplots(figsize=(9,8))#, sharex=True, sharey=True)
    plt.subplots_adjust(left=0.1, right=0.87, top=0.95, bottom=0.1)
    plt.ioff()

    # plot the field
    if profile=="2d":
        ploton.plot(ax, vmin=vmin, vmax=vmax)
        if streamlines:
            vr = vx1on.data
            vphi = vx2on.data
            if isPlanet:
                vphi -= vx2on.omegaplanet*vx2on.xmed[:,None,None]
            if stype=="lic":
                streams=streamon.get_lic_streams(vr,vphi)
                streamon.plot_lic(ax,streams,cartesian=cart, cmap='gray', alpha=0.3)
            elif stype=="random":
                streams=streamon.get_random_streams(vr,vphi,xmin=srmin,xmax=srmax, n=nstream)
                streamon.plot_streams(ax,streams,cartesian=cart,color='k', linewidth=2, alpha=0.5)
            elif stype=="fixed":
                streams=streamon.get_fixed_streams(vr,vphi,xmin=srmin,xmax=srmax, n=nstream)
                streamon.plot_streams(ax,streams,cartesian=cart,color='k', linewidth=2, alpha=0.5)

        if config['midplane']:
            plt.savefig("sRphi_log%s_c%s%04d.png"%(log,cart,on))
        else:
            plt.savefig("sRz_log%s_c%s%04d.png"%(log,cart,on))

    # plot the 1D profile
    if profile=="1d":
        ploton.axiplot(ax, vmin=vmin, vmax=vmax)
        plt.savefig("saxi_log%s%04d.png"%(log,on))

    plt.close()

    if pbar:
        if parallel:
            global counter
            printProgressBar(counter.value, len(config['onarray'])-1, prefix = 'Progress:', suffix = 'Complete', length = 50) # progress bar when parallelization is included
            with counter.get_lock():
                counter.value += 1  # incrementation of the counter
        else:
            printProgressBar(on-config['onarray'][0], len(config['onarray'])-1, prefix = 'Progress:', suffix = 'Complete', length = 50)

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    # parser.add_argument('-dir', action="store", dest="dir", default="")

    # read the .toml file
    # analysis = AnalysisNonos(directory=args.dir)
    analysis = AnalysisNonos(directory="")
    pconfig=analysis.config
    n_file=analysis.n_file
    diran=analysis.directory

    parser.add_argument('-info', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('-mod', type=str, default=pconfig['mode'], help="default: pconfig['mode']")
    parser.add_argument('-on', type=int, default=pconfig['onStart'], help="default: pconfig['onStart']")
    parser.add_argument('-f', type=str.lower, default=pconfig['field'], help="default: pconfig['field']")
    parser.add_argument('-onend', type=int, default=pconfig['onEnd'], help="default: pconfig['onEnd']")
    parser.add_argument('-diff', type=bool, nargs='?', const=True, default=pconfig['diff'], help="default: pconfig['diff']")
    parser.add_argument('-log', type=bool, nargs='?', const=True, default=pconfig['log'], help="default: pconfig['log']")
    parser.add_argument('-cor', type=bool, nargs='?', const=True, default=pconfig['corotate'], help="default: pconfig['corotate']")
    parser.add_argument('-s', type=bool, nargs='?', const=True, default=pconfig['streamlines'], help="default: pconfig['streamlines']")
    parser.add_argument('-stype', type=str, default=pconfig['streamtype'], help="default: pconfig['streamtype']")
    parser.add_argument('-srmin', type=float, default=pconfig['rminStream'], help="default: pconfig['rminStream']")
    parser.add_argument('-srmax', type=float, default=pconfig['rmaxStream'], help="default: pconfig['rmaxStream']")
    parser.add_argument('-sn', type=int, default=pconfig['nstream'], help="default: pconfig['nstream']")
    parser.add_argument('-isp', type=bool, nargs='?', const=True, default=pconfig['isPlanet'], help="default: pconfig['isPlanet']")
    parser.add_argument('-cart', type=bool, nargs='?', const=True, default=pconfig['cartesian'], help="default: pconfig['cartesian']")
    parser.add_argument('-p', type=str, default=pconfig['profile'], help="default: pconfig['profile']")
    parser.add_argument('-cmap', type=str, default=pconfig['cmap'], help="default: pconfig['cmap']")
    parser.add_argument('-full', type=bool, nargs='?', const=True, default=pconfig['fullfilm'], help="default: pconfig['fullfilm']")
    parser.add_argument('-pbar', type=bool, nargs='?', const=True, default=pconfig['progressBar'], help="default: pconfig['progressBar']")
    parser.add_argument('-multi', type=bool, nargs='?', const=True, default=pconfig['parallel'], help="default: pconfig['parallel']")
    parser.add_argument('-cpu', type=int, default=pconfig['nbcpu'], help="default: pconfig['nbcpu']")
    args = parser.parse_args(argv)

    args.f=args.f.upper()
    # plt.close('all')
    if args.info:
        AnalysisNonos(directory="", info=True)

    # mode for just displaying a field for a given output number
    if args.mod=='display':
        fig, ax=plt.subplots(figsize=(9,8))#, sharex=True, sharey=True)
        plt.ioff()
        # print("on = ", args.on)
        # loading the field
        ploton = PlotNonos(pconfig, field=args.f, on=args.on, diff=args.diff, log=args.log, corotate=args.cor, directory=diran)
        if args.s:
            streamon=StreamNonos(pconfig, field=args.f, on=args.on, directory=diran)
            vx1on = FieldNonos(pconfig, field='VX1', on=args.on, diff=False, log=False, corotate=False, directory=diran)
            vx2on = FieldNonos(pconfig, field='VX2', on=args.on, diff=False, log=False, corotate=False, directory=diran)

        # plot the field
        if args.p=="2d":
            ploton.plot(ax, cartesian=args.cart, cmap=args.cmap)
            if args.s:
                vr = vx1on.data
                vphi = vx2on.data
                if args.isp:
                    vphi -= vx2on.omegaplanet*vx2on.xmed[:,None,None]
                if args.stype=="lic":
                    streams=streamon.get_lic_streams(vr,vphi)
                    streamon.plot_lic(ax,streams,cartesian=args.cart, cmap='gray', alpha=0.3)
                elif args.stype=="random":
                    streams=streamon.get_random_streams(vr,vphi,xmin=args.srmin,xmax=args.srmax, n=args.sn)
                    streamon.plot_streams(ax,streams,cartesian=args.cart,color='k', linewidth=2, alpha=0.5)
                elif args.stype=="fixed":
                    streams=streamon.get_fixed_streams(vr,vphi,xmin=args.srmin,xmax=args.srmax, n=args.sn)
                    streamon.plot_streams(ax,streams,cartesian=args.cart,color='k', linewidth=2, alpha=0.5)

        # plot the 1D profile
        if args.p=="1d":
            ploton.axiplot(ax)

        plt.show()

    # mode for creating a movie of the temporal evolution of a given field
    elif args.mod=='film':
        # do we compute the full movie or a partial movie given by "on"
        if args.full:
            pconfig['onarray']=range(n_file)
        else:
            pconfig['onarray']=np.arange(args.on,args.onend+1)

        # calculation of the min/max
        if args.diff:
            vmin=pconfig['vmin']
            vmax=pconfig['vmax']
        # In that case we choose a file in the middle (len(onarray)//2) and compute the MIN/MAX
        else:
            fieldon = FieldNonos(pconfig, field=args.f, on=pconfig['onarray'][len(pconfig['onarray'])//2], directory=diran, diff=False)
            if args.p=="2d":
                vmin=fieldon.data.min()
                vmax=fieldon.data.max()
            elif args.p=="1d":
                vmin=(np.mean(np.mean(fieldon.data,axis=1),axis=1)).min()
                vmax=(np.mean(np.mean(fieldon.data,axis=1),axis=1)).max()

        # call of the process_field function, whether it be in parallel or not
        start=time.time()
        if args.pbar:
            printProgressBar(0, len(pconfig['onarray'])-1, prefix = 'Progress:', suffix = 'Complete', length = 50) # progress bar when parallelization is included
        if args.multi:
            # determines the minimum between nbcpu and the nb max of cpus in the user's system
            nbcpuReal = min((int(args.cpu),os.cpu_count()))
            pool = Pool(nbcpuReal)   # Create a multiprocessing Pool with a security on the number of cpus
            pool.map(functools.partial(process_field, profile=args.p, field=args.f, cart=args.cart, diff=args.diff, log=args.log, corotate=args.cor, streamlines=args.s, stype=args.stype, srmin=args.srmin, srmax=args.srmax, nstream=args.sn, config=pconfig, vmin=vmin, vmax=vmax, isPlanet=args.isp, pbar=args.pbar, parallel=args.multi, directory=diran), pconfig['onarray'])
            tpara=time.time()-start
            print("time in parallel : %f" %tpara)
        else:
            list(map(functools.partial(process_field, profile=args.p, field=args.f, cart=args.cart, diff=args.diff, log=args.log, corotate=args.cor, streamlines=args.s, stype=args.stype, srmin=args.srmin, srmax=args.srmax, nstream=args.sn, config=pconfig, vmin=vmin, vmax=vmax, isPlanet=args.isp, pbar=args.pbar, parallel=args.multi, directory=diran), pconfig['onarray']))
            tserie=time.time()-start
            print("time in serie : %f" %tserie)

    else:
        print("everything's loaded")

    print('\n')

    return 0
