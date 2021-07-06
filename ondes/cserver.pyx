# cython: nonecheck=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
import cython

cimport numpy as np
import numpy as np
import numpy.random
import scipy.fft
import scipy.signal
import time
import sounddevice as sd
import logging

from cpython cimport bool
from libc.math cimport cos, pow, floor
from libc.stdlib cimport malloc, free

from . import config
from . import core

cdef int SAMPLERATE = <int> config.SAMPLERATE
cdef int A_MIDIKEY = <int> config.A_MIDIKEY
cdef int BASENOTE = <int> config.BASENOTE




class Server(object):

    def __init__(self, data):
        #assert isinstance(data, core.Data)
        #self.data = data
        zerosdata = np.zeros((config.BLOCKSIZE, 2), dtype=config.DTYPE)

        # init time matrix
        self.time_matrix = np.zeros((config.BLOCKSIZE, config.OUTCHANNELS), dtype=config.DTYPE)
        for ichan in range(config.OUTCHANNELS):
            self.time_matrix[:,ichan] = np.arange(config.BLOCKSIZE) + np.random.uniform() * config.SAMPLERATE/config.FMIN
            
        self.lastt = 0
        self.input_spectrum = np.random.uniform(size=config.OUTCHANNELS)
        
        self.fmin, self.fmax = 50, 10000
        assert self.fmin >= config.FMIN, 'min frequency too low'
        assert self.fmax <= config.FMAX, 'max frequency too high'
        
        def callback(outdata, frames, timing, status):

            def get_block(name):
                
                try:
                    
                    channels_matrix = self.time_matrix + self.lastt
                    self.lastt += config.BLOCKSIZE
                    
                    freqs = 10**np.linspace(np.log10(self.fmin), np.log10(self.fmax), config.OUTCHANNELS).reshape((1,config.OUTCHANNELS))
                    noct = np.log(self.fmax/self.fmin) / np.log(2)
                    pink = (np.linspace(1, noct, config.OUTCHANNELS)**(-3))
                    
                    channels_matrix *= 2 * np.pi * freqs * pink  / config.SAMPLERATE
                    
                    channels_matrix *= self.input_spectrum # should be an input matrix
                    channels_matrix = np.cos(channels_matrix)

                    sound = np.mean(channels_matrix, axis=1)
                    
                    data = np.zeros((config.BLOCKSIZE, 2), dtype=config.DTYPE)

                    data[:,0] = sound
                    data[:,1] = data[:,0]
                    #data[:,1] = np.cos(t * 2 * np.pi * f / config.SAMPLERATE)
        
                except core.BufferEmpty:
                    #logging.warn('Buffer is empty: increase buffersize?')
                    data = np.copy(zerosdata)
                except Exception as e:
                    print(e)
                    data = np.copy(zerosdata)
                try:
                    len(data)
                except Exception as e:
                    logging.warn('data reading error: {}'.format(e))
                    data = np.copy(zerosdata)
                return data

            stime = time.time()
            
            assert frames == config.BLOCKSIZE

            # put input
            # try:
            #     self.data.put_block('input', indata[:,0], indata[:,1])
            # except core.BufferFull:
            #     logging.warn('Input buffer full')
                
            # except Exception as e:
            #     logging.warn('Input exception: {}'.format(e))
                
                
            # get output
            
            if status.output_underflow:
                logging.warn('Output underflow: increase blocksize?')
                data = np.copy(zerosdata)
            if status:
                logging.warn('callback status')
                data = np.copy(zerosdata)
            else:
                data = get_block('synth')
                # data += get_block('sequencer')
                # data += get_block('sampler')
                
            data = data.astype(config.DTYPE)
            
            #assert data.shape == outdata.shape, 'data len must match'

            # morphing with input

            #data = utils.morph(indata, data, 0.5)
            
            # send to out channel
            outdata[:] = data

            #print(time.time() - stime, config.BLOCKSIZE/config.SAMPLERATE)
            #self.data.timing_buffers['server_callback_time'].put(time.time() - stime)
            return

        self.stream = sd.OutputStream(
            samplerate=config.SAMPLERATE, blocksize=config.BLOCKSIZE,
            device=self.get_device(), channels=2, dtype=config.DTYPE,
            callback=callback)
            
        self.stream.start()
        
        timeout = config.BLOCKSIZE * config.BUFFERSIZE / config.SAMPLERATE

        while True:
            time.sleep(config.SLEEPTIME)

        
    def __del__(self):
        try:
            self.stream.close()
        except Exception as e:
            pass


    def get_device(self):
        devices = sd.query_devices()
        logging.info('audio devices: \n{}'.format(repr(devices)))
        for device in devices:
            if config.DEVICE in device['name']:
                return device['name']

        logging.warning('device not found switching to default')
        return 'default'
