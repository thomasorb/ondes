import numpy as np
import os
import logging

DEVICE = 'Steinberg UR22mkII'
#DEVICE = 'front'
#MIDIDEVICE = 'Steinberg UR22mkII'

MIDIDEVICES = ('MPK mini 3', 'nanoKONTROL2', 'Akai MPD24', 'Steinberg UR22mkII')

#BUFFERSIZE = 10 # number of blocks used for buffering
#BLOCKSIZE = 384 #768 # block size # warning BLOCKSIZE is not really used anymore
#BLOCKTIME = BLOCKSIZE / SAMPLERATE * 1000.
#LATENCY = BLOCKTIME * BUFFERSIZE

LATENCY = 50 # ms
SAMPLERATE = 44100
BUFFERSIZE = int(LATENCY / 1000 * SAMPLERATE) + 1
NCHANNELS = 64 # number of most significant channels


logging.info('sample rate: {} Hz'.format(SAMPLERATE))
logging.info('latency: {:.2f} ms'.format(LATENCY))
logging.info('buffersize: {} frames'.format(BUFFERSIZE))


DTYPE = np.float32
COMPLEX_DTYPE = np.complex64
SLEEPTIME = LATENCY / 10000.

#MAX_SAMPLES = 3 # samples + max notes played per synth
#MAX_SAMPLE_LEN = int(16 * SAMPLERATE / BLOCKSIZE) # in blocks
#logging.info('max sample length: {:.2f} s'.format(MAX_SAMPLE_LEN * BLOCKTIME / 1000.))


MAX_SYNTHS = 1
#SYNTH_LOOP_TIME = 3 # s
#TRANSIT_TIME = SAMPLERATE * SYNTH_LOOP_TIME # in samples

MAX_ATTACK_TIME = 100. # in ms.
MAX_RELEASE_TIME = 1000. # in ms.
VOLUME_ADJUST = 10.
VELOCITY_SCALE = 50. # impact of velocity on note volume, the higher the higher the sound on low velocity
BASENOTE = 60
A_MIDIKEY = 45
PADNOTE_SHIFT = -36
POLYPHONY_VOLUME_ADJUST = 8 # volume of each note is divided by 15 to avoid audio > 1 when multpiple notes are stacked together

TRANS_SIZE = 5
TRANS_RELEASE = 10 # s


MAX_DISPLAY_SIZE = 20000
BINNING = 4 # binning of data vs image
RANDOMWALKER_RADIUS = 3

FMIN = 20 # min frequency in Hz
FMAX = 20000 # max frequency in Hz

# define CC inputs and their defaults values
CC_MATRIX = {
    'freqmin': (0,70,0),
    'freqrange': (0,71,127),
    'srate': (0,72,63),
    'bright': (0,73,0),
    'comp_threshold': (0,74,0),
    'comp_level': (0,75,0),
    'attack_time': (0,76,64),
    'release_time': (0,77,64),
    'volume': (0,200,63),
    'trans_presence': (0,1,0),
    'trans_release': (0,2,64),
    'rec': (0,45,0),
    'keep': (0,41,0),
    'unkeep': (0,42,0)
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
        
