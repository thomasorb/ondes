import numpy as np
import re
import scipy.fft
import hashlib
import time

def fastinterp1d(a, x):
    assert np.all((0 <= x)  * (x <= len(a) - 1))
    xint = np.copy(x).astype(int)
    alow = a[xint]
    xint_high = xint + 1
    xint_high[xint + 1 >= len(a) - 1] = len(a) - 1
    ahigh = a[xint_high]
    return alow + (ahigh - alow) * (x - xint)

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


def read_msg(msg, n):
    durations = {
        'O':32,
        'oo':16, # white
        'oo.':24, # white
        'o':8, # black
        'o.':12, 
        'p':4, # croche
        'p.':6,
        'pp':2, # double croche
        'pp.':3,
        'ppp':1 # triple croche
    }
    arr = np.zeros(n, dtype=bool)

    if 'end' in msg:
        return arr
    
    index = 0
    
    if 'x' in msg:
        for c in list(msg):
            
            if c == 'x':
                arr[index] = True
                index += 1
            elif c == '.':
                index += 1
            elif c == '|':
                index += 8
                index -= index%8
            elif c == ':':
                index += 4
                index -= index%4
            
            
            if index >= n: break
        return arr
    
    redur = re.compile(r'[Oop.*$]+')
    durs = redur.findall(msg)
    index = 0
    for idur in durs:
        silence = False
        if '*' in idur:
            idur = idur.replace('*', '')
            silence = True
        if idur not in durations:
            raise Exception('duration not understood {}'.format(idur))
        if not silence:
            arr[index] = True
        index += durations[idur]
        if index >= n: break
    return arr

def read_synth_msg(msg, n):

        
    renote = re.compile(r'[\dABCDEFG#?]+')
    

    arr = -np.ones(n, dtype=int)
    arrd = np.zeros(n, dtype=int)

    if 'end' in msg:
        return arr, arrd
    index = 0

    notestart = False
    last_note_index = 0
    i = 0
    while not i == len(msg) - 1:
        c = msg[i]
        i += 1
        
        if c=='.':
            if notestart:
                arrd[last_note_index] = index - last_note_index
                
            notestart = False
            index += 1
            
        elif c == '|':
            index += 8
            index -= index%8
        elif c == ':':
            index += 4
            index -= index%4
        elif c == '=':
            index += 1        
        elif renote.findall(c) != []:
            inote = c
            if not notestart:
                notestart = True
                last_note_index = int(index)
            else: 
                arrd[last_note_index] = index - last_note_index
                last_note_index = int(index)
                
            while renote.findall(msg[i]) != []:
                c = msg[i]
                inote += c
                i += 1
            arr[last_note_index] = get_note_value(inote)
                        
        else:
            raise Exception('bad msg formatting')
    if notestart:
        arrd[last_note_index] = index - last_note_index

                

    #for idur, inote in zip(durs, notes):
    #    get_note_value(inote)
    return arr, arrd

def note2f(note, a_midikey):
    return 440. / 2**(float(a_midikey - note) / 12.) / 2.

def note2shift(note, basenote, a_midikey):
    return max(note2f(basenote, a_midikey) / note2f(note, a_midikey), 1)

def sine(f, n, srate):
    x = np.arange(n, dtype=np.float32) / srate
    return np.cos(f * x * 2. * np.pi)

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
