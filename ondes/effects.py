import numpy as np
#import scipy.interpolate
import scipy.fft
import scipy.stats

from . import utils
from . import config
from . import maths

def normalize(data, perc):
    NORM_LEVEL = 0.90
    data /= np.max(data)
    data = np.clip(data, -1, 1)
    print(np.percentile(np.abs(data), perc))
    return compress(data, np.percentile(np.abs(data), perc), NORM_LEVEL)

def expand(data, threshold, log_level):
    assert 0 < threshold < 1
    assert 0 < log_level < 10
    data = maths.expand(data, threshold, log_level)
    return data
    
def compress(data, threshold, level):
    assert 0 < threshold < 1
    assert 0 < level < 1
    data = maths.compress(data, threshold, level)
    return data

def delay(data, delta, coeff, n):
    assert coeff > 0, 'coeff must be > 0'
    delta = int(delta)
    n = int(n)
    
    out = np.zeros((len(data) + n*delta, data.shape[1]), dtype=data.dtype)
    out[:len(data)] = data
    for i in range(n):
        out[(i+1)*delta:(i+1)*delta + len(data)] += (coeff**(i+1)) * data
    return out

def resample(data, samplef):
    if np.isscalar(samplef):
        samplef = np.linspace(0, len(data) - 1, int(len(data) * samplef))
    #samplef = np.clip(samplef, 0, len(data))
    samplef[np.isnan(samplef)] = 0
    samplef = samplef % (len(data) - 1)
        
    new_data = list()
    for ich in range(data.shape[1]):
        #f = scipy.interpolate.UnivariateSpline(
        #    np.arange(len(data)),
        #    data[:,ich], k=3, ext=1, s=0)
        #new_data.append(f(samplef))
        new_data.append(utils.fastinterp1d(data[:,ich], samplef))
    return np.array(new_data).T

def shift(data, note):
    fratio = utils.note2f(0, config.A_MIDIKEY) / utils.note2f(note, config.A_MIDIKEY)
    return resample(data, fratio)

def adsr(data, a, d, s, r):
    new_data = list()
    for ich in range(data.shape[1]):
        new_data.append(data[:,ich] * utils.envelope(len(data), a, d, s, r))
    return np.array(new_data).T

def math(data, op, v):
    v = np.atleast_1d(v)
    sz = len(data)
    if len(v) > 1:
        if len(v.shape) == 1:
            v = np.array((v,v)).T
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


def cut_to_blocksize(sample, blocksize):
    final_size = int(len(sample) // blocksize) * blocksize
            
    if final_size > len(sample):
        ratio = (final_size // len(sample) + 1)
        sample = np.concatenate(list([sample,]) * ratio)
        
    if final_size <= len(sample):
        sample = sample[:final_size]
            
    cut_len = blocksize/final_size
    sample = adsr(sample, cut_len,0,0,cut_len)
    return sample

def crop(sample, start, end):
    assert 0 <= start < 1
    assert 0 < end <= 1
    assert start < end
    return sample[int(len(sample) * start):int(len(sample) * end)]
