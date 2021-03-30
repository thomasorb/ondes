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


MAX_SYNTHS = 1
SYNTH_LOOP_TIME = 4 # s
TRANSIT_TIME = SAMPLERATE * SYNTH_LOOP_TIME / 2.# in samples

VOLUME_ADJUST = 10.
BASENOTE = 102
A_MIDIKEY = 45
PADNOTE_SHIFT = -36



MAX_DISPLAY_SIZE = 20000
BINNING = 4

# pulseaudio -k && sudo alsa force-reload
# ffmpeg -ac 2 -i 'Kick 03.aif' 'Kick 03.wav'


