# cython: nonecheck=False
cimport cython
import cython

cimport numpy as np
import numpy as np
import numpy.random
import scipy.fft
import scipy.signal
import time

from cpython cimport bool
from libc.math cimport cos, pow, floor
from libc.stdlib cimport malloc, free

from . import config

cdef int SAMPLERATE = <int> config.SAMPLERATE
cdef int A_MIDIKEY = <int> config.A_MIDIKEY
cdef int BASENOTE = <int> config.BASENOTE

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef float * downsample(float * a_in, int a_in_size, int note, int basenote) nogil:
#     cdef int i
#     cdef int n = <int> (<float> a_in_size * note2shift(note, basenote))
#     cdef float *a_out = <float *> malloc(n * sizeof(float))
#     cdef float x, c
#     for i in range(n):
#         x = <float>i / <float>(n-1) * <float> (a_in_size - 1)
#         c = x - <float> floor(x)
#         a_out[i] = a_in[<int> x] * (1. - c) + a_in[<int> x + 1] * c
#     return a_out

@cython.boundscheck(False)
@cython.wraparound(False)
def get_first_zero(np.float32_t[::1] c):
    cdef int i
    with nogil:
        for i in range(0, c.shape[0] - 1):
            if c[i] > 0 and c[i+1] < 0:
                break
    return i

@cython.boundscheck(False)
@cython.wraparound(False)
def get_last_zero(np.float32_t[::1] c):
    cdef int i
    with nogil:
        for i in range(c.shape[0] - 2, 0, -1):
            if c[i] > 0 and c[i+1] < 0:
                break
    return i

@cython.boundscheck(False)
@cython.wraparound(False)
cdef min_along_z(np.complex64_t[:,:,::1] a):
    
    cdef int n = a.shape[2]
    cdef np.complex64_t[::1] b = np.ascontiguousarray(np.empty(n, dtype=np.complex64))
    cdef int i, j, k, si, sj
    cdef complex minval
    si = a.shape[0]
    sj = a.shape[1]
    with nogil:
        for ik in range(n):
            minval = a[0,0,ik]
            for ii in range(si):
                for ij in range(sj):
                    if a[ii,ij,ik].real < minval.real:
                        minval = a[ii,ij,ik]
            b[ik] = minval
    return b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mean_along_z(np.complex64_t[:,:,::1] a):
    
    cdef int n = a.shape[2]
    cdef np.complex64_t[::1] b = np.ascontiguousarray(np.empty(n, dtype=np.complex64))
    cdef int i, j, k, si, sj
    cdef complex _sum
    cdef float _min, _max, _val
    si = a.shape[0]
    sj = a.shape[1]
    cdef complex area = <complex> (si * sj)
    with nogil:
        for ik in range(n):
            _sum = 0
            _min = abs(a[0,0,ik])
            _max = abs(a[0,0,ik])
            for ii in range(si):
                for ij in range(sj):
                    _val = abs(a[ii,ij,ik])
                    _sum += a[ii,ij,ik]
                    if _val < _min: _min = _val
                    if _val > _max: _max = _val
            b[ik] = b[ik] - <complex> (_min + _max)
            b[ik] = _sum / (area - 2.)
    return b


def sine(float f, int n, int srate):
    cdef np.ndarray[np.float32_t, ndim=1] x = np.arange(n, dtype=np.float32) / <float> srate
    return np.cos(f * x * 2. * np.pi)

def square(float f, int n, int srate):
    return np.sign(sine(f, n, srate))

@cython.cdivision(True)
cdef float note2f(int note) nogil:
    return 440. / pow(2.,((<float> A_MIDIKEY - <float> note) / 12.)) / 2.

@cython.cdivision(True)
cdef float note2shift(int note, int basenote) nogil:
    return max(note2f(basenote) / note2f(note), 1)

@cython.cdivision(True)
def release_values(double time, double stime, double rtime, int buffersize):
    cdef double sv, ev, deltat
    global SAMPLERATE
    deltat = rtime - time + stime
    sv = deltat / rtime
    ev = (deltat - <double> buffersize / <double> SAMPLERATE) / rtime
    ev = max(0, ev)
    return sv, ev

@cython.cdivision(True)
def attack_values(double time, double stime, double atime, int buffersize):
    cdef double sv, ev, deltat
    global SAMPLERATE
    deltat = time - stime
    sv = deltat / atime
    sv = min(1, max(0, sv))
    ev = (deltat + <double> buffersize / <double> SAMPLERATE) / atime
    ev = min(1, ev)
    return sv, ev

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef forward_transform(np.float32_t[:] x, int dirty=500):
    """from scipy.signal.resample code: https://github.com/scipy/scipy/blob/master/scipy/signal/signaltools.py"""
    cdef int N = x.shape[0]
    # Forward transform
    cdef np.complex64_t[::1] X = np.ascontiguousarray(scipy.fft.rfft(x, overwrite_x=True))
    cdef np.float32_t[::1] W = np.ascontiguousarray(scipy.fft.ifftshift(
        scipy.signal.get_window(('gaussian', dirty), N))).astype(np.float32)
    cdef int i
    # Fold the window back on itself to mimic complex behavior
    with nogil:
        for i in range(1, X.shape[0]):
            W[i] = (W[i] + W[W.shape[0] - i]) * 0.5
            X[i] = X[i] * W[i]
        X[0] = X[0] * W[0]
            
    # Copy positive frequency components (and Nyquist, if present)
    X[N // 2 + 1:] = 0.

    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    if N % 2 == 0:
        # select the component at frequency +N/2 and halve it
        X[N//2] = X[N//2] * 0.5
    return X

@cython.cdivision(True)
def inverse_transform(np.complex64_t[::1] X, int note, int basenote):
    cdef int N = X.shape[0]
    N = (N - N%2) * 2

    cdef double shift = note2shift(note, basenote)
    cdef int num = <int> (<double> N * shift)
    
    num = scipy.fft.next_fast_len(num)
    
    # Inverse transform
    cdef float norm = <float> num / <float> N
    return np.ascontiguousarray(
        scipy.fft.irfft(X, num, overwrite_x=True).real * norm).astype(np.float32)
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_buffer(np.float32_t[::1] a, int index, int N):
    cdef np.float32_t[::1] buf = np.ascontiguousarray(np.empty(N, dtype=np.float32))
    cdef int i
    with nogil:
        for i in range(N):
            buf[i] = a[index]
            index += 1
            if index > a.shape[0] - 1:
                index = 0
    return buf, index
            
        
cdef class Wave(object):

    cdef int indexL, indexR
    cdef np.float32_t[::1] sampleL
    cdef np.float32_t[::1] sampleR
    cdef np.complex64_t[::1] base_sampleL
    cdef np.complex64_t[::1] base_sampleR
     
    cdef int NL, NR, Nbase
    cdef str mode
    cdef int dirty
    cdef double time
    cdef bool lock
    
    def __init__(self, str mode='sine', int dirty=500):
        self.indexL = 0
        self.indexR = 0
        self.Nbase = scipy.fft.next_fast_len(4000)
        self.dirty = dirty
        self.mode = mode
        self.base_sampleL = self.compute_base_sample(True)
        self.base_sampleR = self.compute_base_sample(False)
        

    def get_base_samples(self):
        return np.array(self.base_sampleL), np.array(self.base_sampleR)
    
    # def get_samples(self, int note):
    #     global BASENOTE
        
    #     self.lock = True
    #     self.sampleL = self.inverse_transform(self.base_sampleL, note, BASENOTE)
    #     self.sampleR = self.inverse_transform(self.base_sampleR, note, BASENOTE)

    #     self.sampleL = self.sampleL[get_first_zero(self.sampleL):get_last_zero(self.sampleL)]
    #     self.sampleR = self.sampleR[get_first_zero(self.sampleR):get_last_zero(self.sampleR)]
        
    #     self.NL = self.sampleL.shape[0]
    #     self.NR = self.sampleR.shape[0]
        
    #     self.lock = False
    #     return np.array(self.sampleL), np.array(self.sampleR)
    
    cdef compute_base_sample(self, int left):
        global SAMPLERATE
        global BASENOTE
                    
        cdef np.float32_t[:] s
        if self.mode == 'square':
            s = square(note2f(BASENOTE),  self.Nbase, SAMPLERATE)
        else:
            s = sine(note2f(BASENOTE),  self.Nbase, SAMPLERATE)
        return forward_transform(s, self.dirty)

    # cdef inverse_transform(self, np.complex64_t[::1] X, int note, int basenote):
    #     return inverse_transform(X, note, basenote)
    
        
    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    # cdef get_buffers(self, int BUFFERSIZE, int velocity,
    #                 float sv=1, float ev=1, float volume=0.5):

    #     cdef int i
    #     cdef float ienv
    #     cdef np.float32_t[::1] bufL
    #     cdef np.float32_t[::1] bufR

    #     while self.lock: time.sleep(0.000001)
    #     bufL, self.indexL = get_buffer(self.sampleL, self.indexL, BUFFERSIZE)
    #     bufR, self.indexR = get_buffer(self.sampleR, self.indexR, BUFFERSIZE)
    #     with nogil:
    #         for i in range(BUFFERSIZE):
    #             ienv = <float> i / <float> BUFFERSIZE * (ev - sv) + sv
    #             ienv *= <float> velocity / 64. * volume
    #             bufL[i] = bufL[i] * ienv 
    #             bufR[i] = bufR[i] * ienv
                
    #     return bufL, bufR
    

# cdef class DataWave(Wave):

#     cdef np.complex64_t[:,:,::1] data
#     cdef int iiL, ijL, iiR, ijR
#     cdef int tune
    
#     def __init__(self, np.complex64_t[:,:,::1] data, int note, int posx, int posy,
#                  int dirty=500, int tune=36):
#         self.data = data
#         self.iiL = posx
#         self.ijL = posy
#         self.iiR = posx
#         self.ijR = posy
#         self.tune = tune
        
#         Wave.__init__(self, note, mode='sine', dirty=dirty)

#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     @cython.cdivision(True)
#     cdef get_base_sample(self, int left, bool init=True):
#         global INTEGSIZE
#         cdef int ii, ij
#         cdef np.complex64_t[::1] base_sample
#         cdef int i
#         cdef int RSIZE = 5

#         if left:
#             self.iiL = random_move(self.iiL, RSIZE, 0, self.data.shape[0])
#             self.ijL = random_move(self.ijL, RSIZE, 0, self.data.shape[1])
#             ii = self.iiL
#             ij = self.ijL
            
#         else:
#             self.iiR = random_move(self.iiR, RSIZE, 0, self.data.shape[0])
#             self.ijR = random_move(self.ijR, RSIZE, 0, self.data.shape[1])
#             ii = self.iiR
#             ij = self.ijR

#         base_sample = mean_along_z(self.data[ii:ii+INTEGSIZE,
#                                              ij:ij+INTEGSIZE, :])
#         if init: return base_sample
        
#         with nogil:
#             for i in range(base_sample.shape[0]):
#                 if left == 1:
#                     base_sample[i] = (base_sample[i] + self.base_sampleL[i]) / 2.
#                 else:
#                     base_sample[i] = (base_sample[i] + self.base_sampleR[i]) / 2.
#         return base_sample

#     cdef inverse_transform(self, np.complex64_t[::1] X, int note, int basenote):
#         cdef np.float32_t[:] y = inverse_transform(X, note, basenote + self.tune)
#         cdef int border = max(<int> (<float> y.shape[0] * 0.1), 10)
#         return y[border:y.shape[0]//2 - <int>(y.shape[0]*0.125)]
    

# def data2view(np.ndarray[np.complex64_t, ndim=3] data):
#     cdef np.complex64_t[:,:,::1] view
#     view = data
#     return view

# cdef random_move(int ii, int s, int imin, int imax):
#    ii += np.random.randint(-s,s)
#    imin = imin + s + 1
#    imax = imax - s - 1
#    if ii < imin: ii = imin
#    if ii > imax - 1: ii = imax - 1
#    return ii
   
            
    


