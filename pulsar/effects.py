import numpy as np
import scipy.interpolate
import scipy.fft

from . import utils

def delay(data, delta, coeff, n):
    assert coeff > 0, 'coeff must be > 0'
    delta = int(delta)
    n = int(n)
    
    out = np.zeros((len(data) + n*delta, data.shape[1]), dtype=data.dtype)
    out[:len(data)] = data
    for i in range(n):
        out[(i+1)*delta:(i+1)*delta + len(data)] += (coeff**(i+1)) * data
    return out

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

def math(data, op, v):
    v = np.atleast_1d(v)
    sz = len(data)
    if len(v) > 1:
        sz = min(len(data), len(v))
    
    if op == 'mult':
        return data[:sz] * v[:sz]
    elif op == 'add':
        return data[:sz] + v[:sz]
    else: raise Exception('unknown operation')

def mult(data, v):
    return math(data, 'mult', v)

def add(data, v):
    return math(data, 'add', v)

def filter(data, window, *args):
    if isinstance(window, str):
        window = getattr(utils, window)(len(data), *args)
    else:
        assert window.size == len(data), 'window must have same size as data'
    data_fft = scipy.fft.fft(data, axis=0)
    data_fft *= window.reshape((len(window), 1))
    return scipy.fft.ifft(data_fft, axis=0).real




