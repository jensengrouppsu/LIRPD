import numpy as np
from raman_density import RamanDensityObj
from constants import ANGSTROM2BOHR
from utils import construct_grid, plot_ters_image
from time import time
import os, sys
from glob import glob

'''
An example code that runs the LIRPD method on a molecule's alpha and A-tensor densities.

'''

#########################
#  Part I: Integration  #
#########################



# Choose how many processes to use in the integration for speeding up
nproc = 4

# Use any cube file for initialization
cube_init = 'example-data/benzene/mode835.94-alpha-real-zz.cube'

# Define field magnitude
mr, mi = 25, 25
# Define near field confinement
b = 2.5  # angstrom
bx, by, bz = b, b, b/2.0
br = ANGSTROM2BOHR(np.array([bx,by,bz]))
bi = ANGSTROM2BOHR(np.array([bx,by,bz]))
# Define focal plane height
height = ANGSTROM2BOHR(0.8)



# Initialization for integration
d = RamanDensityObj(cubefile=cube_init)
height += d.coords[:,2].max()
xpara = (d.coords[:,0].min()-5, d.coords[:,0].max()+5, 1)
ypara = (d.coords[:,1].min()-5, d.coords[:,1].max()+5, 1)
nx = int((xpara[1]-xpara[0])/xpara[2])+1
ny = int((ypara[1]-ypara[0])/ypara[2])+1

# Construct scanning grid. Store coordinates in scanxyz
gridinfo = {}
gridinfo['org'] = [xpara[0], ypara[0], height]
gridinfo['stepsize'] = [xpara[2], ypara[2], 0]
gridinfo['nstep'] = [nx, ny, 1]
scanxyz = construct_grid(gridinfo)

# Store withs and magnitude parameters for the field distribution
d.field_params['mr'] = mr
d.field_params['br'] = br
d.field_params['mi'] = mi
d.field_params['bi'] = bi

# Perform local integration of Raman polarizability densities
outfile = 'example-data/benzene/mode835.94.integrated.pickle' # path to the integrated density

zzr = 'example-data/benzene/mode835.94-alpha-real-zz.cube'
zzzr = 'example-data/benzene/mode835.94-aatensor-real-zzz.cube'
zzxr = 'example-data/benzene/mode835.94-aatensor-real-zzx.cube'
zzyr = 'example-data/benzene/mode835.94-aatensor-real-zzy.cube'
d.collect_alpha_cube(zzr=zzr)
print("Alpha-tensor collected!")
d.collect_aatensor_cube(zzzr=zzzr, zzxr=zzxr, zzyr=zzyr)
print("Aa-tensor collected!")

# Scan the field while integrating the densities
print('Integrating...')
v_freq=835.94   # Need v_freq for this mode to calculate cross-section
d.scan_constant_height(scanxyz, v_freq=v_freq, part='both', out=outfile, nproc=nproc)



### Part II: Plotting ###

print('Plotting...')
outfile = 'example-data/benzene/mode835.94.png'    
integrated = np.load('example-data/benzene/mode835.94.integrated.pickle')
scanxyz = scanxyz.reshape(nx,ny,3)  # Reshape the scan coordinates for plotting
plot_ters_image(scanxyz[:,:,0], scanxyz[:,:,1], integrated.reshape(nx,ny), d.coords, outfile)
