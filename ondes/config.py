import numpy as np
import os
import logging

DEVICE = 'Steinberg UR22mkII'
MIDIDEVICE = 'Steinberg UR22mkII'
#MIDIDEVICE = 'Akai MPD24'

BUFFERSIZE = 30 # number of blocks used for buffering
BLOCKSIZE = 768 #768 # block size

SAMPLERATE = 44100
BLOCKTIME = BLOCKSIZE / SAMPLERATE * 1000.

logging.info('sample rate: {} Hz'.format(SAMPLERATE))
logging.info('latency: {:.2f} ms'.format(BLOCKTIME))

DTYPE = np.float32
COMPLEX_DTYPE = np.complex64
SLEEPTIME = BLOCKTIME / 1000. / 10.

MAX_SAMPLES = 3 # samples + max notes played per synth
MAX_SAMPLE_LEN = int(16 * SAMPLERATE / BLOCKSIZE) # in blocks
logging.info('max sample length: {:.2f} s'.format(MAX_SAMPLE_LEN * BLOCKTIME / 1000.))


MAX_SYNTHS = 1
#SYNTH_LOOP_TIME = 3 # s
#TRANSIT_TIME = SAMPLERATE * SYNTH_LOOP_TIME # in samples

VOLUME_ADJUST = 10.
BASENOTE = 102
A_MIDIKEY = 45
PADNOTE_SHIFT = -36

MAX_DISPLAY_SIZE = 20000
BINNING = 4


FMIN = 20 # min frequency in Hz
FMAX = 20000 # max frequency in Hz


CC_SHIFT = 'cc16'
CC_BITS = 'cc17'
CC_V0 = 'cc18'
CC_V1 = 'cc19'
CC_V2 = 'cc20'
CC_SOFTNESS = 'cc21'


CC_IN = {
    '1':0,
    '2':127,
    '3':63,
    '4':0,
    '5':0,
    '6':127,
    '7':63,
    '8':127,
    '33':0,
    '34':0,
    '35':0,
    '40':63,
    '89':0,
    }

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
NCHANNELS = 1 # data treated as mono
NNPATH = '.brain.pth'
        
