import numpy as np
import re
import scipy.fft
import hashlib
import time
from . import ccore

def power_spectrum(s, srate, coeff=2):
    n = len(s)
    axis = scipy.fft.fftfreq(coeff*n, 1/srate)[:int(n*coeff/2)]
    pows = np.abs(scipy.fft.fft(s, n=coeff*n)[:int(n*coeff/2)])
    return axis, pows

def bpfilter(s, fmin, fmax, srate, width=100):
    # width in Hz
    n = len(s)
    axis = np.abs(scipy.fft.fftfreq(n, 1/srate))
    
    sfft = scipy.fft.fft(s, n=n)

    w = ((axis > fmin) * (axis < fmax)).astype(float)
    rmin = (axis > fmin - np.pi * width) * (axis < fmin)
    w[rmin] = (np.cos((axis[rmin] - fmin)/width) + 1) /2.
    rmax = (axis < fmax + np.pi * width) * (axis > fmax)
    w[rmax] = (np.cos((axis[rmax] - fmax)/width) + 1) /2.
    return scipy.fft.ifft(sfft * w)

def fastinterp1d(a, x):
    """may interpolate 2D array along first axis"""
    assert np.all((0 <= x)  * (x <= len(a) - 1))
    xint = np.copy(x).astype(int)
    alow = a[xint]
    xint_high = xint + 1
    xint_high[xint + 1 >= len(a) - 1] = len(a) - 1
    ahigh = a[xint_high]
    if a.ndim == 1:
        return alow + (ahigh - alow) * np.array((x - xint), dtype=a.dtype)
    elif a.ndim == 2:
        return alow + (ahigh - alow) * np.array((x - xint).reshape((x.size,1)), dtype=a.dtype)
    else: raise Exception('array must be 1d or 2d')

def envelope(n, a, d, s, r):
    x = np.arange(n, dtype=float) / (n-1)
    env = np.ones_like(x)
    if a > 0:
        env *= np.clip(x/a, 0, 1)
    if d > 0:
        env *= np.clip((1.-s) * (1.-((x-a)/d)), 0, (1-s)) + s
    if r > 0:
        env *= np.clip((-x+1)/r, 0, 1)
    return env

    
def lopass(n, coeff):
    assert n%2 == 0, 'n must be even'
    w = np.ones(n, dtype=np.float32)
    w[int(n * coeff/2):] = 0
    w[n//2:] = w[:n//2][::-1]
    return w

def hipass(n, coeff):
    return 1. - lopass(n, coeff)

def bdpass(n, center, width):
    return lopass(n, (center+width)) * hipass(n, (center-width))

def note2f(note, a_midikey):
    return 440. / 2**((a_midikey - note) / 12.) / 2.

def note2shift(note, basenote, a_midikey):
    return max(note2f(basenote, a_midikey) / note2f(note, a_midikey), 1)

def sine(f, n, srate, phase=0):
    x = np.arange(n, dtype=np.float32) / srate
    return np.cos(f * (x + phase) * 2. * np.pi)

def square(f, n, srate):
    return np.sign(sine(f, n, srate))

def inverse_transform(X, note, basenote, a_midikey):
    N = X.shape[0]
    N = (N - N%2) * 2

    shift = note2shift(note, basenote, a_midikey)
    num = int(N * shift)
    
    num = scipy.fft.next_fast_len(num)
    
    # Inverse transform
    norm = num /  float(N)
    return np.ascontiguousarray(
        scipy.fft.irfft(X, num, overwrite_x=True).real * norm).astype(np.float32)

def get_hash():
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode())
    return np.array(list(hash.hexdigest()[:10].encode())).astype(np.int32)

def get_note_name(note):
    notes = ['C ', 'C#', 'D ', 'D#', 'E ', 'F ', 'F#', 'G ', 'G#', 'A ', 'A#', 'B ']    
    return '{}{}'.format(notes[note%12], note//12)

def get_note_value(note):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    reoct = re.compile(r'[\d?]+')
    renote = re.compile(r'[ABCDEFG#?]+')

    octave = int(reoct.findall(note)[0])
    note = renote.findall(note)[0]
    return octave*12 + notes.index(note)

def innerpad(a, n):
    assert n >= 2*a.shape[-1]
    inn_shape = list(a.shape)
    inn_shape[-1] = n
    inn = np.zeros(inn_shape, dtype=a.dtype)
    x = np.arange(0, n, n//a.shape[-1])[:a.shape[-1]]
    inn[..., x] = a[..., :x.size]
    return inn

def spec2sample(spec, duration, samplerate, minfreq=None, maxfreq=None, reso=1):

    length = int(duration * samplerate)
    assert length >= 2 * spec.size
    freqstep = samplerate / length / 2
    
    endfreq = freqstep * (length - 1)
    
    if minfreq is None: minfreq = 0
    minpix = minfreq / freqstep
    
    if maxfreq is not None:
        assert maxfreq <= endfreq
        assert maxfreq > minfreq
        maxpix = maxfreq / freqstep
        assert maxpix - minpix >= 2 * spec.size
    else:
        maxpix = minpix + spec.size
            
    spec.real = normalize_spectrum(spec.real, reso)
    spec.imag = normalize_spectrum(spec.imag, reso)
    
    if maxfreq is not None:
        padded_size = int(maxpix - minpix)
        spec = innerpad(spec, padded_size)
        
    padl = int(minpix)
    padr = int(length - minpix - spec.size)# + length
    
    spec = np.pad(spec, ((padl, padr)))
    
    specfft = scipy.fft.fft(spec)

    # not optimal but working perfectly
    specfft = bpfilter(specfft, 20, 20000, samplerate, width=20)
    
    
    sample = np.array([specfft.real[:],
                       specfft.imag[:]]).T
    
    return sample

def normalize_spectrum(spec, reso):
    sign = np.sign(spec)
    spec = np.abs(spec)
    if np.all(spec == 0):
        return spec
    spec -= np.percentile(spec, 1)
    spec /= np.percentile(spec, 99.9)
    spec = np.clip(spec, 0, 1)
    spec = spec**reso
    spec *= sign
    #spec = orb.utils.vector.smooth(spec, 2)
    return spec


def equalize_spectrum(spec, factor):
    # linspace 1 to 10 for 10 octaves
    # factor of -3 should give pink noise but unsure
    # http://hyperphysics.phy-astr.gsu.edu/hbase/Audio/equal.html
    return spec * np.linspace(1, 10, len(spec))**factor


def bit_crush(a, bits, binning):
    if bits < 32:
        a = ccore.dither(a, bits)
    if binning > 1:
        a = ccore.reduce_srate(a, binning)
    return a

def cc_rescale(cc, minscale, maxscale, invert=False):
    if not invert:
        return cc / 127. * (maxscale - minscale) + minscale
    else:
        return (1 - cc / 127.) * (maxscale - minscale) + minscale

def morph(a1, a2, mix):
    return scipy.fft.ifft(
        scipy.fft.fft(a1, axis=0) * mix
        + scipy.fft.fft(a2, axis=0) * (1 - mix), axis=0)


def cut(s, srate, starttime, stoptime):
    if starttime >= stoptime: raise Exception('starttime >= stoptime')
    if starttime < 0: raise Exception('starttime < 0')
    n = len(s)
    start = int(starttime * srate)
    stop = int(stoptime * srate) + 1
    if start >= n: raise Exception('start time exceed sample length')
    if stop >= n: stop = n-1
    return s[start:stop,:]
    
    
    
