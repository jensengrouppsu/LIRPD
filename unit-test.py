import numpy as np
from raman_density import RamanDensityObj
from time import time
from constants import ANGSTROM2BOHR
import sys

if __name__ == "__main__":

    nproc = int(sys.argv[1])

    #test = RamanDensityObj(cubefile='./test-cotpp/mode3115.162-zz-alpha.real.cube')
    #test.collect_alpha_density(zzr='./test-cotpp/mode3115.162-zz-alpha.real.cube')
    #start = time()
    #xpara = ANGSTROM2BOHR(np.array([-9,9.36, 0.36]))
    #ypara = ANGSTROM2BOHR(np.array([-9,9.36, 0.36]))
    #print(np.arange(-9,9.36, 0.36).shape)
    #height = 2
    #test.field_params['br'] = ANGSTROM2BOHR(np.array([2,2,1]))
    #test.scan_constant_height(xpara, ypara, height, nproc=nproc)
    #print("Time for scan_constant_height: {:.3f}s".format(time()-start))

    #test = RamanDensityObj(cubefile='./test-benzene/mode835.94-zz-alpha.real.cube')
    #test.collect_alpha_cube(zzr='./test-benzene/mode835.94-zz-alpha.real.cube')
    #start = time()
    #xpara = ANGSTROM2BOHR(np.array([-4,4,0.32]))
    #ypara = ANGSTROM2BOHR(np.array([-4,4,0.32]))
    #print(np.arange(-4, 4.32, 0.32).shape)
    #height = 2
    #test.field_params['br'] = ANGSTROM2BOHR(np.array([1.5, 1.5, 1.5/2.0]))
    #test.scan_constant_height(xpara, ypara, height, v_freq=835.94, nproc=nproc, out='test-benzene/mp.pickle')
    #print("Time for scan_constant_height: {:.3f}s".format(time()-start))

    test = RamanDensityObj(
        cubefile='./test-benzene/mode835.94-zz-alpha.real.cube')
    test.collect_density_pickle('./test-pickle/mode835.94-alpha.pickle')
    start = time()
    xpara = ANGSTROM2BOHR(np.array([-4, 4, 0.4]))
    ypara = ANGSTROM2BOHR(np.array([-4, 4, 0.4]))
    height = ANGSTROM2BOHR(1.0)
    test.field_params['br'] = ANGSTROM2BOHR(np.array([1.5, 1.5, 1.5/2.0]))
    test.scan_constant_height(xpara, ypara, height, v_freq=835.94, nproc=nproc,
                              out='test-pickle/mode835.94-alpha.mp.pickle')
    print("Time for scan_constant_height: {:.3f}s".format(time()-start))
