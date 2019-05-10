import numpy as np
from local_integration import scan_tip
from utils import construct_grid, read_cube, cross_section


class RamanDensityObj(object):
    def __init__(self, cubefile=None):
        from os import environ

        #os.environ['OMP_NUM_THREADS'] = '2'
        #environ['OMP_DYNAMIC'] = 'TRUE'
        #environ['MKL_DYNAMIC'] = 'TRUE'
        environ['MKL_NUM_THREADS'] = '1'

        if cubefile is None:
            exit("A cube file is required for initialzation!")
        # Initialize object with a cube file containing the density of one tensor element.
        try:
            cube = open(cubefile, 'r')
        except IOError:
            exit("Error loading {}".format(cubefile))

        cube.readline()  # Ignore first line
        cube.readline()  # Ignore second line

        gridinfo = {'org': [0, 0, 0],       # Vector of box origin / Bohr
                    'nstep': [0, 0, 0],     # Number of steps in each direction
                    'stepsize': [0, 0, 0]}  # Step size in each direction / Bohr
        l = cube.readline().strip().split()
        natm, gridinfo['org'] = int(l[0]), map(float, l[1:])
        l = cube.readline().strip().split()
        gridinfo['nstep'][0], gridinfo['stepsize'][0] = int(l[0]), float(l[1])
        l = cube.readline().strip().split()
        gridinfo['nstep'][1], gridinfo['stepsize'][1] = int(l[0]), float(l[2])
        l = cube.readline().strip().split()
        gridinfo['nstep'][2], gridinfo['stepsize'][2] = int(l[0]), float(l[3])
        gridinfo['ngrid'] = gridinfo['nstep'][0] * \
            gridinfo['nstep'][1]*gridinfo['nstep'][2]
        self.gridinfo = gridinfo
        self.cubegrid = construct_grid(gridinfo)

        atm = np.zeros(natm, dtype=int)
        coords = np.zeros((natm, 3), dtype=float)
        for i in range(natm):
            l = cube.readline().strip().split()
            atm[i], coords[i] = int(l[0]), map(float, l[2:])
        self.natoms = natm
        self.atoms = atm
        self.coords = np.array(coords)
        cube.close()

        self.field_params = {
            'direction': 2,
            'mr': 25,           # Atomic unit
            'br': [2, 2, 1],    # Angstrom
            'mi': None,         # Atomic unit
            'bi': None          # Angstrom
        }
        self.alpha = None
        self.aatensor = None

        print("Initialization done!")

    def collect_density_pickle(self, alpha=None, aatensor=None):
        if alpha is not None:
            tensors = np.load(alpha)
            self.alpha = tensors['alpha']
            shape = self.alpha.shape
            if len(shape) > 3:
                self.alpha = self.alpha.reshape(-1, shape[-2], shape[-1])
        if aatensor is not None:
            tensors = np.load(aatensor)
            self.aatensor = tensors['aatensor']
            shape = self.aatensor.shape
            if len(shape) > 4:
                self.aatensor = self.aatensor.reshape(-1, shape[-3], shape[-2], shape[-1])


    def collect_alpha_cube(self, xxr=None, yyr=None, zzr=None, xyr=None, xzr=None, yzr=None,
                           xxi=None, yyi=None, zzi=None, xyi=None, xzi=None, yzi=None):
        '''
        A function to collect polarizability density from cubefiles.
        The alpha density tensor's dimension is : (ngrid, 3, 3).
        Args:
            xyr-> /path/to/xy-component.real.cube
            xyi-> /path/to/xy-component.real.cube
            ...
        Returns:
            N/A
        '''
        self.alpha = np.zeros((self.gridinfo['ngrid'], 3, 3))

        if xxr is not None:
            self.alpha[:, 0, 0].real = read_cube(xxr, self.natoms)
        if yyr is not None:
            self.alpha[:, 1, 1].real = read_cube(yyr, self.natoms)
        if zzr is not None:
            self.alpha[:, 2, 2].real = read_cube(zzr, self.natoms)
        if xyr is not None:
            self.alpha[:, 0, 1].real = read_cube(xyr, self.natoms)
        if xzr is not None:
            self.alpha[:, 0, 2].real = read_cube(xzr, self.natoms)
        if yzr is not None:
            self.alpha[:, 1, 2].real = read_cube(yzr, self.natoms)

        if xxi is not None:
            self.alpha[:, 0, 0].imag = read_cube(xxi, self.natoms)
        if yyi is not None:
            self.alpha[:, 1, 1].imag = read_cube(yyi, self.natoms)
        if zzi is not None:
            self.alpha[:, 2, 2].imag = read_cube(zzi, self.natoms)
        if xyi is not None:
            self.alpha[:, 0, 1].imag = read_cube(xyi, self.natoms)
        if xzi is not None:
            self.alpha[:, 0, 2].imag = read_cube(xzi, self.natoms)
        if yzi is not None:
            self.alpha[:, 1, 2].imag = read_cube(yzi, self.natoms)

        self.alpha[:, 1, 0] = self.alpha[:, 0, 1]
        self.alpha[:, 2, 0] = self.alpha[:, 0, 2]
        self.alpha[:, 2, 1] = self.alpha[:, 1, 2]

    def collect_aatensor_cube(self,
                              xxxr=None, xyyr=None, xzzr=None, xxyr=None, xxzr=None, xyzr=None,
                              yxxr=None, yyyr=None, yzzr=None, yxyr=None, yxzr=None, yyzr=None,
                              zxxr=None, zyyr=None, zzzr=None, zxyr=None, zxzr=None, zyzr=None,
                              xxxi=None, xyyi=None, xzzi=None, xxyi=None, xxzi=None, xyzi=None,
                              yxxi=None, yyyi=None, yzzi=None, yxyi=None, yxzi=None, yyzi=None,
                              zxxi=None, zyyi=None, zzzi=None, zxyi=None, zxzi=None, zyzi=None):
        '''
        A function to collect polarizability density from cubefiles.
        Aatensor density's dimension is (ngrid, 3, 3, 3)
        Args:
            xyzr -> /path/to/xyz-component.real.cube
            xyzi -> /path/to/xyz-component.real.cube
            ...
        Returns:
            N/A
        '''
        self.aatensor = np.zeros((self.gridinfo['ngrid'], 3, 3, 3))

        if xxxr is not None:
            self.aatensor[:, 0, 0, 0].real = read_cube(xxxr, self.natoms)
        if xyyr is not None:
            self.aatensor[:, 0, 1, 1].real = read_cube(xyyr, self.natoms)
        if xzzr is not None:
            self.aatensor[:, 0, 2, 2].real = read_cube(xzzr, self.natoms)
        if xxyr is not None:
            self.aatensor[:, 0, 0, 1].real = read_cube(xxyr, self.natoms)
            self.aatensor[:, 0, 1, 0].real = self.aatensor[:, 0, 0, 1].real
        if xxzr is not None:
            self.aatensor[:, 0, 0, 2].real = read_cube(xxzr, self.natoms)
            self.aatensor[:, 0, 2, 0].real = self.aatensor[:, 0, 0, 2].real
        if xyzr is not None:
            self.aatensor[:, 0, 1, 2].real = read_cube(xyzr, self.natoms)
            self.aatensor[:, 0, 2, 1].real = self.aatensor[:, 0, 1, 2].real

        if xxxi is not None:
            self.aatensor[:, 0, 0, 0].imag = read_cube(xxxi, self.natoms)
        if xyyi is not None:
            self.aatensor[:, 0, 1, 1].imag = read_cube(xyyi, self.natoms)
        if xzzi is not None:
            self.aatensor[:, 0, 2, 2].imag = read_cube(xzzi, self.natoms)
        if xxyi is not None:
            self.aatensor[:, 0, 0, 1].imag = read_cube(xxyi, self.natoms)
            self.aatensor[:, 0, 1, 0].imag = self.aatensor[:, 0, 0, 1].imag
        if xxzi is not None:
            self.aatensor[:, 0, 0, 2].imag = read_cube(xxzi, self.natoms)
            self.aatensor[:, 0, 2, 0].imag = self.aatensor[:, 0, 0, 2].imag
        if xyzi is not None:
            self.aatensor[:, 0, 1, 2].imag = read_cube(xyzi, self.natoms)
            self.aatensor[:, 0, 2, 1].imag = self.aatensor[:, 0, 1, 2].imag

        if yxxr is not None:
            self.aatensor[:, 1, 0, 0].real = read_cube(xxxr, self.natoms)
        if yyyr is not None:
            self.aatensor[:, 1, 1, 1].real = read_cube(xyyr, self.natoms)
        if yzzr is not None:
            self.aatensor[:, 1, 2, 2].real = read_cube(xzzr, self.natoms)
        if yxyr is not None:
            self.aatensor[:, 1, 0, 1].real = read_cube(xxyr, self.natoms)
            self.aatensor[:, 1, 1, 0].real = self.aatensor[:, 1, 0, 1].real
        if yxzr is not None:
            self.aatensor[:, 1, 0, 2].real = read_cube(xxzr, self.natoms)
            self.aatensor[:, 1, 2, 0].real = self.aatensor[:, 1, 0, 2].real
        if yyzr is not None:
            self.aatensor[:, 1, 1, 2].real = read_cube(xyzr, self.natoms)
            self.aatensor[:, 1, 2, 1].real = self.aatensor[:, 1, 1, 2].real

        if yxxi is not None:
            self.aatensor[:, 1, 0, 0].imag = read_cube(xxxi, self.natoms)
        if yyyi is not None:
            self.aatensor[:, 1, 1, 1].imag = read_cube(xyyi, self.natoms)
        if yzzi is not None:
            self.aatensor[:, 1, 2, 2].imag = read_cube(xzzi, self.natoms)
        if yxyi is not None:
            self.aatensor[:, 1, 0, 1].imag = read_cube(xxyi, self.natoms)
            self.aatensor[:, 1, 0, 1].imag = self.aatensor[:, 1, 0, 1].imag
        if yxzi is not None:
            self.aatensor[:, 1, 0, 2].imag = read_cube(xxzi, self.natoms)
            self.aatensor[:, 1, 2, 0].imag = self.aatensor[:, 1, 0, 2].imag
        if yyzi is not None:
            self.aatensor[:, 1, 1, 2].imag = read_cube(xyzi, self.natoms)
            self.aatensor[:, 1, 2, 1].imag = self.aatensor[:, 1, 1, 2].imag

        if zxxr is not None:
            self.aatensor[:, 2, 0, 0].real = read_cube(xxxr, self.natoms)
        if zyyr is not None:
            self.aatensor[:, 2, 1, 1].real = read_cube(xyyr, self.natoms)
        if zzzr is not None:
            self.aatensor[:, 2, 2, 2].real = read_cube(xzzr, self.natoms)
        if zxyr is not None:
            self.aatensor[:, 2, 0, 1].real = read_cube(xxyr, self.natoms)
            self.aatensor[:, 2, 1, 2].real = self.aatensor[:, 2, 0, 1].real
        if zxzr is not None:
            self.aatensor[:, 2, 0, 2].real = read_cube(xxzr, self.natoms)
            self.aatensor[:, 2, 2, 0].real = self.aatensor[:, 2, 0, 2].real
        if zyzr is not None:
            self.aatensor[:, 2, 1, 2].real = read_cube(xyzr, self.natoms)
            self.aatensor[:, 2, 2, 1].real = self.aatensor[:, 2, 1, 2].real

        if zxxi is not None:
            self.aatensor[:, 2, 0, 0].imag = read_cube(xxxi, self.natoms)
        if zyyi is not None:
            self.aatensor[:, 2, 1, 1].imag = read_cube(xyyi, self.natoms)
        if zzzi is not None:
            self.aatensor[:, 2, 2, 2].imag = read_cube(xzzi, self.natoms)
        if zxyi is not None:
            self.aatensor[:, 2, 0, 1].imag = read_cube(xxyi, self.natoms)
            self.aatensor[:, 2, 1, 0].imag = self.aatensor[:, 2, 0, 1].imag
        if zxzi is not None:
            self.aatensor[:, 2, 0, 2].imag = read_cube(xxzi, self.natoms)
            self.aatensor[:, 2, 2, 0].imag = self.aatensor[:, 2, 0, 2].imag
        if zyzi is not None:
            self.aatensor[:, 2, 1, 2].imag = read_cube(xyzi, self.natoms)
            self.aatensor[:, 2, 2, 1].imag = self.aatensor[:, 2, 1, 2].imag

    def scan_constant_height(self, scanxyz, v_freq=0, laser=634, temperature=298, part='both',
                             out="out.pickle", nproc=1):
        '''
        Define a scanning plane (assuming XY plane)
            scanxyz -> the grid coordinates of the scanning plane.
            height -> height of the scanning plane.
            v_freq -> normal mode frequency in Wavenumbers.
            laser -> laser wavelength in Nanometers.
            temperature -> assumed temperature in K for the calculation of cross sections.
            out -> dumping cross-sections to file "out" (pickle).
            nproc -> for multiprocessing. default is 1 for sequential processing.
        Need to define self.field_params that will be passed to local_integraion.scan_tip.
        '''

        self.field_params.update({'cubegrid': self.cubegrid})

        # Construct scanning plane
        #minX, maxX, stepX = xpara
        #minY, maxY, stepY = ypara
        #nx = int((maxX-minX)/stepX + 1) 
        #ny = int((maxY-minY)/stepY + 1) 
        #gridinfo = {}
        #gridinfo['org'] = [minX, minX, height]
        #gridinfo['nstep'] = [nx, ny, 1]
        #gridinfo['stepsize'] = [stepX, stepY, 0]
        #scanxyz = construct_grid(gridinfo)

        dr = reduce(lambda x, y: x*y, self.gridinfo['stepsize'])
        # Multiprocessing the scanning
        if part == 'imag':
            self.alpha.real = self.alpha.real * 0
            if self.aatensor is not None:
                self.aatensor.real = self.aatensor.real * 0
        elif part == 'real':
            self.alpha.imag= self.alpha.imag* 0
            if self.aatensor is not None:
                self.aatensor.imag = self.aatensor.imag * 0
        self.aD = scan_tip(self.alpha, self.aatensor,
                           scanxyz, self.field_params, nproc)*dr
        self.aD = cross_section(self.aD, v_freq, laser, temperature)
        #self.aD = self.aD.reshape(nx, ny)
        self.aD.dump(out)
