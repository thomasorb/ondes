import sounddevice as sd
import logging
import multiprocessing as mp
import queue
import numpy as np
import time
import scipy.interpolate

from . import core
from . import config
from . import utils

class Server(object):

    def __init__(self, data, file_path, file_srate, invert_channels=False):
        #assert isinstance(data, core.Data)
        #self.data = data

        # if 2D, must have shape (samples, channels)
        self.filedata = np.load(file_path).astype(config.DTYPE)
        logging.info('data shape: {}'.format(self.filedata.shape))
        logging.info('data type: {}'.format(self.filedata.dtype))

        self.interp_factor = config.SAMPLERATE / file_srate * 4 # * 4 beware hack
        
        if invert_channels:           
            self.filedata = self.filedata[:,::-1]
            
        self.filedata /= np.max(self.filedata)
        self.filedata = self.filedata ** 5
        
        lp_factor = 4
        window = scipy.signal.get_window(('gaussian', lp_factor), lp_factor*4)
        window = window.reshape((window.size, 1))

        self.filedata = scipy.signal.fftconvolve(
            self.filedata, window, mode='same', axes=0)
        

        self.filedata = self.filedata[:self.filedata.shape[0] - (self.filedata.shape[0] % config.BLOCKSIZE),:]
        logging.info('data shape (cut to blocksize): {}'.format(self.filedata.shape))
        
        
        
        self.nchans = int(self.filedata.shape[1])


        zerosdata = np.zeros((config.BLOCKSIZE, 2), dtype=config.DTYPE)
            
        self.last_index = 0
        #self.input_spectrum = np.random.uniform(low=0.5, high=1, size=self.nchans).reshape((1, self.nchans))
        
        self.fmin, self.fmax = 50, 10000
        assert self.fmin >= config.FMIN, 'min frequency too low'
        assert self.fmax <= config.FMAX, 'max frequency too high'
        
        # will be recomputed async
        self.freqs = 10**np.linspace(np.log10(self.fmin), np.log10(self.fmax), self.nchans).reshape((1, self.nchans))
        noct = np.log(self.fmax/self.fmin) / np.log(2) # number of octaves
        self.pink = (np.linspace(1, noct, self.nchans)**(-3))

        # init time matrix
        self.time_matrix = np.zeros((config.BLOCKSIZE, self.nchans), dtype=config.DTYPE)
        for ichan in range(self.nchans):
            self.time_matrix[:,ichan] = np.arange(config.BLOCKSIZE) + np.random.uniform() * config.SAMPLERATE / self.freqs[:,ichan]
            
        def callback(outdata, frames, timing, status):

            def get_block():
                
                try:
                    channels_matrix = np.copy(self.time_matrix) + self.last_index
                    
                    channels_matrix *= 2 * np.pi * self.freqs / config.SAMPLERATE 

                    channels_matrix = np.cos(channels_matrix)

                    data_index = self.last_index #% self.filedata.shape[0]
                    data_index /= self.interp_factor
                    interp_axis = np.arange(config.BLOCKSIZE, dtype=float) / self.interp_factor + data_index
                    interp_axis = interp_axis % (self.filedata.shape[0] - 1)
                    
                    filedata_block = utils.fastinterp1d(
                        self.filedata, interp_axis)

                    channels_matrix *= filedata_block

                    #channels_matrix *= self.filedata[self.last_index:self.last_index+config.BLOCKSIZE,:]
                    
                    
                    channels_matrix *= self.pink
                    
                    sound = np.mean(channels_matrix, axis=1)
                    
                    data = np.zeros((config.BLOCKSIZE, 2), dtype=config.DTYPE)
                    data[:,0] = sound
                    data[:,1] = data[:,0]
                    
                except Exception as e:
                    logging.warn('callback exception: {}'.format(e))
                    data = np.copy(zerosdata)

                self.last_index += config.BLOCKSIZE
                    
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
                data = get_block()
                
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
                print(device)
                return device['name']

        logging.warning('device not found switching to default')
        return 'default'
