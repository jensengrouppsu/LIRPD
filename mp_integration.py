import numpy as np
from LIRPD import RamanDensityObj
from LIRPD.constants import ANGSTROM2BOHR
from LIRPD.utils import construct_grid
from time import time
import os, sys
from glob import glob

'''
An example code that runs the LIRPD method on a molecule's alpha and A-tensor densities.

'''

def local_intengration():
    # Initialization
    d = RamanDensityObj(cubefile=cube_init)
    xpara = (d.coords[:,0].min()-5, d.coords[:,0].max()+5, 1.0)
    ypara = (d.coords[:,1].min()-5, d.coords[:,1].max()+5, 1.0)
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

    # Looping over all modes and perform local integration of Raman polarizability densities
    n_freqs = len(freqs)
    for i in range(n_freqs):
        v_freq = freqs[i]
        print('[{} {} {} {}] {}/{} Mode{}:'.format(part, mol, bx, h, i+1, n_freqs, v_freq))

        print '  Collecting density pickle...',
        start = time()
        p_alpha= '/path/to/mode*.alpha.pickle' # path to the alpha-tensor density
        p_aatensor= '/path/to/mode*.aatensor.pickle' # path to the A-tensor density
        outfile = '/path/to/output.pickle' # path to the integrated density
        if os.path.isfile(outfile):
            print(outfile)
            continue
         
        if comp == 'alpha':
            d.collect_density_pickle(p_alpha)
        elif comp == 'aatensor':
            d.collect_density_pickle(p_alpha, p_aatensor)
        else:
            d.collect_density_pickle(None, p_aatensor)
        print "done: {:.3f}s".format(time()-start) 

        print '  Integrating...',
        start = time()

        # Need v_freq for this mode to calculate cross-section
        try:
            v_freq = float(v_freq)
        except ValueError:
            v_freq = float(v_freq.split('_')[0])+0.001
        d.scan_constant_height(scanxyz, v_freq=v_freq, part=part, out=outfile, nproc=nproc)
        print "done: {:.3f}s".format(time()-start)

if __name__ == '__main__':

    global br           # Width for field real part
    global bi           # Width for field imag part
    global mr           # Magnitude for field real part
    global mi           # Magnitude for field imag part
    global height       # Focal-plane height
    global nproc        # Number of processes to use 
    global cube_init    # A .cube file for initialization
    global freqs        # Frequencies of vibrational modes
    global comp         # Components to use (alpha, or alpha+A)
    global part         # Real or complex part to use

    # Decide with tensor to include:
    # 'alpha' -> alpha density only
    # 'aatensor' -> alpha + Aatensor densities
    comp = 'aaonly'
    comp = 'alpha'
    comp = 'aatensor'

    part = 'imag'
    part = 'real'
    part = 'both'

    if part == 'imag':
        mr, mi = 0, 25
    elif part == 'real':
        mr, mi = 25, 0
    else:
        mr, mi = 25, 25

    # Choose how many processes to use in the integration for speeding up
    nproc = 8

    pickles = glob( "/path/to/mode*.alpha*.pickle")

    freqs = []
    for p in pickles:
        freqs.append(p.partition('mode')[-1].partition('-')[0])
    freqs.sort()
    freqs = np.array(freqs)
    print(freqs)

    cube_init = '/path/to/mode*.density.cube' # Use any cube file for initialization

    # Define near field confinement
    b = 6.0  # angstrom
    bx, by, bz = b, b, b/2.0
    br = ANGSTROM2BOHR(np.array([bx,by,bz]))
    bi = ANGSTROM2BOHR(np.array([bx,by,bz]))

    # Define focal plane height
    height = ANGSTROM2BOHR(3.1)

    try:
        os.makedirs("/path/to/integrated/")
    except OSError:
        pass

    local_intengration()

