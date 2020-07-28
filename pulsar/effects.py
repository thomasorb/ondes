import numpy as np
import scipy.interpolate

from . import utils

def delay(data, delta, coeff, n):
    delta = int(delta)
    n = int(n)
    assert coeff < 1, 'coeff must be < 1'
    assert coeff > 0, 'coeff must be > 0'
    final_size = int(delta * (n+1))
    if len(data) < final_size:
        print('appe')
        data = np.concatenate((data, np.zeros((final_size - len(data), data.shape[1]))))
    print(data.shape, final_size)
    for i in range(n):
        data[(i+1) * delta:(i+2) * delta] += coeff * data[i * delta:(i+1) * delta]
    return data

def resample(data, coeff):
    new_data = list()
    for ich in range(data.shape[1]):
        new_data.append(scipy.interpolate.UnivariateSpline(
            np.arange(len(data)),
            data[:,ich], k=3, ext=1, s=0)(np.linspace(0, len(data) - 1, int(len(data) * coeff))))
    return np.array(new_data).T

def adsr(data, a, d, s, r):
    new_data = list()
    for ich in range(data.shape[1]):
        new_data.append(data[:,ich] * utils.envelope(len(data), a, d, s, r))
    return np.array(new_data).T

