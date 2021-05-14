import sounddevice as sd
import logging
import multiprocessing as mp
import queue
import numpy as np
import time

from . import core
from . import config
from . import utils

class Server(object):

    def __init__(self, data):
        assert isinstance(data, core.Data)
        self.data = data
        zerosdata = np.zeros((config.BLOCKSIZE, 2), dtype=config.DTYPE)
        
        def callback(indata, outdata, frames, timing, status):

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
            
            assert data.shape == outdata.shape, 'data len must match'

            # morphing with input

            #data = utils.morph(indata, data, 0.5)
            
            # send to out channel
            outdata[:] = data

            self.data.timing_buffers['server_callback_time'].put(time.time() - stime)
            return

        self.stream = sd.Stream(
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
