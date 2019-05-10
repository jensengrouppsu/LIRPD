import numpy as np


def construct_grid(gridinfo):
    '''
    Returns a grid matrix of shape (ngrid, 3)
    '''
    orgx, orgy, orgz = gridinfo['org']
    nx, ny, nz = gridinfo['nstep']
    stepx, stepy, stepz = gridinfo['stepsize']
    x = np.linspace(orgx, orgx+(nx-1)*stepx, nx)
    y = np.linspace(orgy, orgy+(ny-1)*stepy, ny)
    z = np.linspace(orgz, orgz+(nz-1)*stepz, nz)
    gx, gy, gz = np.meshgrid(x, y, z, indexing='ij')
    gx = gx.flatten()
    gy = gy.flatten()
    gz = gz.flatten()
    return np.stack((gx, gy, gz)).transpose()


def read_cube(cubefile, natoms):
    '''
    A function to collect density values from a cube file. Headers are descarded.
    '''
    try:
        cube = open(cubefile, 'r')
    except IOError:
        exit("Error loading {}".format(cubefile))

    # Skip the headers and atomic coordinates
    for i in range(6+natoms):
        cube.readline()
    data = []
    for line in cube:
        data.extend(line.split())
    cube.close()
    return np.array(data, dtype=float)


def read_coordinates(xyzfile):
    try:
        return np.loadtxt(xyzfile, skiprows=2, usecols=(1, 2, 3), dtype=float)
    except IOError:
        exit("Error loading {}".format(xyzfile))


def cross_section(aD, v_freq, laser=634, temperatur=298):
    from constants import PI, PLANCK, LIGHT, AMU, BOLTZMAN
    from constants import NM2WAVENUM, WAVENUM2INVM, M2CM, BOHR2ANGSTROM
    '''
    Convert Raman polarizability into Raman cross section.
    Args:
        freq:   normal frequencies in Wavenumbers
        aD:     Raman polarizability tensor (nfreq, 3, 3)
        laser:  incident light frequency in NM
        TEMPERATURE: default is 298 K
    '''
    assert v_freq, "Provide normal mode frequency in Wavenumber"
    CONVERSION = 2 * PI**2 * PLANCK * 1E-40 / (LIGHT * AMU)
    EXPARG = PLANCK * LIGHT / BOLTZMAN
    lambda_0 = NM2WAVENUM(laser)
    boltzfact = (1 - np.exp(-EXPARG * WAVENUM2INVM(v_freq) / temperatur))
    frequency = WAVENUM2INVM(lambda_0 - v_freq)**4
    # Convert from m^2 to cm^2
    scat = np.absolute((45 * aD.conjugate() * aD) * BOHR2ANGSTROM**4)
    # The cross section is returned in :math:`\\frac{cm^2}{sr}`.
    return M2CM(M2CM((scat * CONVERSION) * frequency/(45 * boltzfact * WAVENUM2INVM(v_freq))))


def plot_ters_image(gridx, gridy, integrated, coords=None, out='out.png'):
    from matplotlib import rcParams
    from matplotlib import pyplot as plt

    params = {
        'axes.labelsize': 20,
        'font.size': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'mathtext.fontset': 'stix',
        #'font.family': 'sans-serif',
        #'font.sans-serif': 'Arial',
        'axes.linewidth': 2.0
    }
    rcParams.update(params)

    fig = plt.figure(figsize=(8.1, 6.5))

    dmin = integrated.min()
    dmax = integrated.max()
    v = np.linspace(dmin, dmax, 100)
    fg = plt.contourf(gridx, gridy, integrated, levels=v, cmap=plt.cm.jet)
    for c in fg.collections:
        c.set_edgecolor("face")
    tick = np.linspace(dmin, dmax, 5)
    plt.colorbar(ticks=tick)
    if coords is not None:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        a = np.argsort(z)
        plt.plot(x[a], y[a], 'wo', markersize=2, mew=2, color='white')
    #plt.xlim((-17.9, 17.9))
    #plt.ylim((-17.9, 17.9))
    fig.savefig(out, transparent=True, dpi=150)
    plt.close()


def lorentzian(x, peak=0, height=1.0, fwhm=None):
    '''Calculates a three-parameter lorentzian for a given domain.'''
    if fwhm is None:
        raise ValueError('lorentzian: fwhm must be given')
    else:
        gamma = fwhm / 2.0
        return (height / np.pi) * (gamma / ((x - peak)**2 + gamma**2))


def sum_lorentzian(x, peak=None, height=None, fwhm=None):
    '''Calculates and sums several lorentzians to make a spectrum.

    'peak' and 'height' are numpy arrays of the peaks and heights that
    each component lorentzian has.
    '''
    from numpy import array
    if peak is None or height is None:
        raise ValueError('Must pass in values for peak and height')
    if peak.shape != height.shape:
        raise ValueError('peak and height must be the same shape')

    y = array([lorentzian(x, peak[i], height[i], fwhm) for i in xrange(len(peak))])
    return y.sum(axis=0)


def plot_ters_spectrum(freqs, integrated, out='out.png', fwhm=20):

    from matplotlib import rcParams
    from matplotlib import pyplot as plt

    params = {
        'axes.labelsize': 16,
        'font.size': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'mathtext.fontset': 'stix',
        #'font.family': 'sans-serif',
        #'font.sans-serif': 'Arial',
        'axes.linewidth': 2.0
    }
    rcParams.update(params)

    fig = plt.figure(figsize=(8.1, 4.5))

    x_smooth = np.linspace(freqs.min()-50, freqs.max()+50, 2000)
    y_smooth = sum_lorentzian(x_smooth, freqs, integrated, fwhm=fwhm)

    plt.plot(x_smooth, y_smooth, 'r')

    stickscale = 1 / ((fwhm / 2) * np.pi)
    plt.stem(freqs, integrated*stickscale, 'k-', 'k ', 'k ')

    plt.xlabel(r"Wavenumbers (cm$^{-1}$)")
    plt.ylabel(r"Cross sections($\frac{{\mathrm{{km}}}}{{\mathrm{{mol}}}}$)")
    plt.tight_layout()
    fig.savefig(out,transparent=True, dpi=150)
    plt.close()
