import numpy as np
from field_models import field_lorentzian
from multiprocessing import Process, Manager, Pool

def scan_tip(alpha, aatensor, scanxyz, field_params, nproc=1):
    '''
    Calculate TERS image by constant-height scanning.
    Args:
        alpha -> alpha densities (ngrid, 3, 3)
        aatensor -> aatensor densities (ngrid, 3, 3, 3)
        scanxyz -> coordinates of tip scan points (nscan, 3)
        field_params -> dict of parameters for field_lorentzian
    Returns:
        aD -> dressed Raman polarizabilities after density integration at all given scan positions.
              (nscan, 3, 3)
    '''

  # manager = Manager()
  # shared_res = manager.list([[]]*nproc)
  # proc = []
  # scanchunk = np.array_split(scanxyz, nproc)

  # def __worker(scanchunk, i):
  #     tmp = manager.list(shared_res[i])
  #     for tip in scanchunk:
  #         field_params.update({'center': tip})
  #         E, fg = field_lorentzian(**field_params)
  #         tmp.append(_dressed_tensors(alpha, aatensor, E, fg))
  #     shared_res[i] = tmp
  #     #return res
  # for i in range(nproc):
  #     p = Process(target=__worker, args=(scanchunk[i], i))
  #     p.daemon = True
  #     proc.append(p)
  #     p.start()
  # for p in proc:
  #     p.join()
  # res = []
  # for i in range(nproc):
  #     res.extend(shared_res[i])
  # print(res)
  # return np.array(res)



    scanchunk = np.array_split(scanxyz, nproc)

    inputs = []
    for i in range(nproc):
        tmp = []
        tmp.append(scanchunk[i])
        tmp.append(alpha)
        tmp.append(aatensor)
        tmp.append(field_params)
        inputs.append(tmp)


    p = Pool(nproc)
    output = p.map(worker, inputs)
    
    res = []
    for i in range(nproc):
        res.extend(output[i])

    return np.array(res)
    
    

def worker(inputs):
    chunk = inputs[0]
    alpha = inputs[1]
    aatensor = inputs[2]
    field_params = inputs[3]
    tmp = []
    for tip in chunk:
        field_params.update({'center':tip})
        E, fg = field_lorentzian(**field_params)
        tmp.append(_dressed_tensors(alpha, aatensor, E, fg))
    return tmp

def _dressed_tensors(alpha, aatensor, E, fg):
    '''
    This function employs the dressed-tensors formula on alpha and A-tensor densities.
    Args:
        alpha -> alpha densities (ngrid, 3, 3)
        aatensor -> Aatensor densities (ngrid, 3, 3, 3)
        E -> near field distribution (ngrid, 3, 3)
        fg -> near field gradient (ngrid, 3, 3, 3)
    '''
    aD = 0.0
    if alpha is not None:
        aD += np.einsum('iac,icd,ibd->iab', E, alpha, E)
    if (aatensor is not None and fg is not None):
        aD += (1. / 3.) * np.einsum('iacd,iecd,ibe->iab', fg, aatensor, E)
    aD = sum(aD[:,2,2])

    return aD
