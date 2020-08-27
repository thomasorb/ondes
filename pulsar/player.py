import sounddevice as sd
import logging
import multiprocessing as mp
import queue
import numpy as np
import time

from . import core
from . import config


class Player(object):

    def __init__(self, data):
        assert isinstance(data, core.Data)
        self.data = data
        zerosdata = np.zeros((config.BLOCKSIZE, 2), dtype=config.DTYPE)
            
        def callback(outdata, frames, time, status):

            def get_block(name):
                try:
                    data = self.data.get_block(name)
                except core.BufferEmpty:
                    #logging.warn('Buffer is empty: increase buffersize?')
                    data = np.copy(zerosdata)
                try:
                    len(data)
                except Exception as e:
                    logging.warn('data reading error: {}'.format(e))
                    data = np.copy(zerosdata)
                return data

            assert frames == config.BLOCKSIZE
            if status.output_underflow:
                logging.warn('Output underflow: increase blocksize?')
                data = np.copy(zerosdata)
            if status:
                logging.warn('callback status')
                data = np.copy(zerosdata)
            else:
                data = get_block('sequencer')
                data += get_block('sampler')
                data += get_block('synth')

            data = data.astype(config.DTYPE)
            
            assert data.shape == outdata.shape, 'data len must match'
            outdata[:] = data

        self.stream = sd.OutputStream(
            samplerate=config.SAMPLERATE, blocksize=config.BLOCKSIZE,
            device=config.DEVICE, channels=2, dtype=config.DTYPE,
            callback=callback)#, finished_callback=self.data.event.set)

        self.stream.start()
        
        timeout = config.BLOCKSIZE * config.BUFFERSIZE / config.SAMPLERATE

        while True:
            time.sleep(config.SLEEPTIME)

        
    def __del__(self):
        try:
            self.stream.close()
        except Exception as e:
            pass 
