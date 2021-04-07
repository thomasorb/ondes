import numpy as np
import os
import logging

DEVICE = 'Steinberg UR22mkII'
MIDIDEVICE = 'Akai MPD24'

BUFFERSIZE = 10 # number of blocks used for buffering
BLOCKSIZE = 512 #768 # block size
SAMPLERATE = 44100
BLOCKTIME = BLOCKSIZE / SAMPLERATE * 1000.

logging.info('sample rate: {} Hz'.format(SAMPLERATE))
logging.info('latency: {:.2f} ms'.format(BLOCKTIME))

DTYPE = np.float32
COMPLEX_DTYPE = np.complex64
SLEEPTIME = BLOCKTIME / 1000. / 100.

MAX_SAMPLES = 3 # samples + max notes played per synth
MAX_SAMPLE_LEN = int(16 * SAMPLERATE / BLOCKSIZE) # in blocks
logging.info('max sample length: {:.2f} s'.format(MAX_SAMPLE_LEN * BLOCKTIME / 1000.))


MAX_SYNTHS = 3
SYNTH_LOOP_TIME = 3 # s
TRANSIT_TIME = SAMPLERATE * SYNTH_LOOP_TIME # in samples

VOLUME_ADJUST = 10.
BASENOTE = 102
A_MIDIKEY = 45
PADNOTE_SHIFT = -36

MAX_DISPLAY_SIZE = 20000
BINNING = 4


CC_SHIFT = 'cc16'
CC_BITS = 'cc17'
CC_V0 = 'cc18'
CC_V1 = 'cc19'
CC_V2 = 'cc20'

CC16_DEFAULT = 0.5
CC17_DEFAULT = 1.
CC18_DEFAULT = 1.
CC19_DEFAULT = 1.
CC20_DEFAULT = 1.
CC21_DEFAULT = 0.

XY = {
    0: (250, 250),
    1: (200, 200),
    2: (300, 300)
}

# pulseaudio -k && sudo alsa force-reload
# ffmpeg -ac 2 -i 'Kick 03.aif' 'Kick 03.wav'


