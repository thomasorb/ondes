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
            assert frames == config.BLOCKSIZE
            if status.output_underflow:
                logging.warn('Output underflow: increase blocksize?')
                data = zerosdata
            #assert not status
            if status:
                data = zerosdata
            else:
                try:
                    data = self.data.get_block()
                except core.BufferEmpty:
                    #logging.warn('Buffer is empty: increase buffersize?')
                    data = zerosdata

                try:
                    len(data)
                except Exception as e:
                    logging.warn('data reading error: {}'.format(e))
                    data = zerosdata

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
