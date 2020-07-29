import numpy as np
import os

TEMPO = 84
BEATS = 4 # number of beats per measure (this is only used to compute tempo, and must not be changed)
STEPS = 32 # number of steps per measure, 32 means that 1 step = 1 triple croche
MAX_SAMPLES = 10
MAX_SAMPLE_LEN = 1000 # in buffers
MAX_STEPS_PER_MEASURE = 64
MAX_INSTRUMENTS = 8

SAMPLES_FOLDER = os.path.split(os.path.abspath(__file__))[0] + os.sep + 'samples' + os.sep

BUFFERSIZE = 30 # number of blocks used for buffering
BLOCKSIZE = 200#768 # block size
SAMPLERATE = 44100
DTYPE = np.float32
SLEEPTIME = BLOCKSIZE / SAMPLERATE / 1000.


DEVICE = 'HDA Intel PCH: ALC293 Analog (hw:0,0)'
#DEVICE = 'Steinberg UR22mkII: USB Audio'

# pulseaudio -k && sudo alsa force-reload
# ffmpeg -ac 2 -i 'Kick 03.aif' 'Kick 03.wav'

