import numpy as np


def field_lorentzian(center=[0, 0, 0], cubegrid=None, direction=2,
                     mr=1, br=[1, 1, 1], mi=0, bi=[1,1,1]):
    '''
    3D Lorentzian field. zz component of the local field tensor is assumed.
    Args:
        center -> the vector of the field center.
        grid -> the dictionary containing grid informations obtained from the cube files.
        direction -> direction of the local field polarization, assuming the same as external field.
        mr, mi -> magnitudes of the real and imaginary parts.
        br, bi -> FWHMs in each Cartesian direction for real and imag parts.
    By providing mi, bxi, byi, bzi, complex field will be returned.
    '''
    assert len(center) == 3, "Make sure all 3 coodirnates are provided for field center."
    assert cubegrid is not None, "Provide the RamanDensityObj.grid dictionary."
    assert len(br) == 3, "Make sure all 3 widths are provided for field distribution."
    N = len(cubegrid)
    field = np.zeros((N, 3, 3), dtype=complex)
    fg = np.zeros((len(cubegrid), 3, 3, 3), dtype=complex)

    if mr > 0:
       #rxyzr = map(lambda x: x/2.0, br)
        rxyzr = [x/2.0 for x in br]
        denominatorR = (np.square((cubegrid - center) / rxyzr).sum(axis=-1) + 1.)
        field[:, direction, direction].real = mr / denominatorR + 1. # Add in the unit external field
        fdxr = -4. * (cubegrid[:, 0] - center[0]) / (br[0] * denominatorR**2)
        fdyr = -4. * (cubegrid[:, 1] - center[1]) / (br[1] * denominatorR**2)
        fdzr = -4. * (cubegrid[:, 2] - center[2]) / (br[2] * denominatorR**2)
        fg[:, direction, direction].real = mr * np.stack((fdxr, fdyr, fdzr)).transpose()

    if mi > 0:
       #rxyzi = map(lambda x: x/2.0, bi)
        rxyzi = [x/2.0 for x in bi]
        denominatorI = (np.square((cubegrid - center) / rxyzi).sum(axis=-1) + 1.)
        field[:, direction, direction].imag = mi / denominatorI
        fdxi = -4. * (cubegrid[:, 0] - center[0]) / (bi[0] * denominatorI**2)
        fdyi = -4. * (cubegrid[:, 1] - center[1]) / (bi[1] * denominatorI**2)
        fdzi = -4. * (cubegrid[:, 2] - center[2]) / (bi[2] * denominatorI**2)
        fg[:, direction, direction].imag = mi * np.stack((fdxi, fdyi, fdzi)).transpose()

    return field, fg


def field_gaussian(center=[0, 0, 0], cubegrid=None, direction=2,
                   mr=1, br=[1, 1, 1], mi=None, bi=None):
    '''
    To be implemented
    '''
    pass
