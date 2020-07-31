import numpy as np
import re
import scipy.fft
import hashlib
import time

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

def note2f(note, a_midikey):
    return 440. / 2**(float(a_midikey - note) / 12.) / 2.

def note2shift(note, basenote, a_midikey):
    return max(note2f(basenote, a_midikey) / note2f(note, a_midikey), 1)

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

        

