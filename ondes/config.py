import numpy as np
import os
import logging

#DEVICE = 'MacBook Pro Speakers'
DEVICE = 'EVO8'
#DEVICE = 'front'
#MIDIDEVICE = 'Steinberg UR22mkII'

MIDIDEVICES = ('nanoKONTROL2', 'Akai MPD24')

BUFFERSIZE = 3 # number of blocks used for buffering
BLOCKSIZE = 256 #768 # block size

SAMPLERATE = 44100
BLOCKTIME = BLOCKSIZE / SAMPLERATE * 1000.
LATENCY = BLOCKTIME * BUFFERSIZE
logging.info('sample rate: {} Hz'.format(SAMPLERATE))
logging.info('latency: {:.2f} ms'.format(LATENCY))

NCHANNELS = 32 # number of most significant channels

DTYPE = np.float32
COMPLEX_DTYPE = np.complex64
SLEEPTIME = BLOCKTIME / 1000. / 10.

MAX_SAMPLES = 3 # samples + max notes played per synth
MAX_SAMPLE_LEN = int(16 * SAMPLERATE / BLOCKSIZE) # in blocks
logging.info('max sample length: {:.2f} s'.format(MAX_SAMPLE_LEN * BLOCKTIME / 1000.))


MAX_SYNTHS = 1
#SYNTH_LOOP_TIME = 3 # s
#TRANSIT_TIME = SAMPLERATE * SYNTH_LOOP_TIME # in samples

MAX_ATTACK_TIME = 100. # in ms.
MAX_RELEASE_TIME = 1000. # in ms.
VOLUME_ADJUST = 10.
VELOCITY_SCALE = 10. # impact of velocity on note volume
BASENOTE = 56
A_MIDIKEY = 45
PADNOTE_SHIFT = -36

MAX_DISPLAY_SIZE = 20000
BINNING = 4


FMIN = 20 # min frequency in Hz
FMAX = 20000 # max frequency in Hz

# define CC inputs and their defaults values
CC_MATRIX = {
    'freqmin': (0,16,0),
    'freqrange': (0,17,127),
    'srate': (0,18,63),
    'bright': (0,19,0),
    'comp_threshold': (0,20,0),
    'comp_level': (0,21,0),
    'attack_time': (0,22,64),
    'release_time': (0,23,64),
    'volume': (0,0,63),
    'rec': (0,45,0)
    }


# CC_IN = { # controller
#     '16':0,
#     '17':127,
#     '18':63,
#     '19':0,
#     '20':0,
#     '21':127,
#     '22':63,
#     '23':127,
#     '0':0,
#     '34':0,
#     '35':0,
#     '40':63,
#     '45':0,
#     }

XY = {
    0: (250, 250),
    1: (200, 200),
    2: (300, 300)
}

# pulseaudio -k && sudo alsa force-reload

# reinstall also and pulseaudio
# sudo dpkg --purge --force-depends pulseaudio alsa-base alsa-utils
# sudo apt --fix-broken install
# ffmpeg -ac 2 -i 'Kick 03.aif' 'Kick 03.wav'


## NN params
#NCHANNELS = 1 # data treated as mono || warning another paramter is called NCHANNELS
#NNPATH = '.brain.pth'
        
