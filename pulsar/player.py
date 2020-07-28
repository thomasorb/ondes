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
        zerosdata = np.zeros((config.BLOCKSIZE, config.CHANNELS), dtype=config.DTYPE)
            
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
                    data = self.data.qget()
                except queue.Empty:
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
            device=config.DEVICE, channels=config.CHANNELS, dtype=config.DTYPE,
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

# event = threading.Event()




# try:

#     with sf.SoundFile(args.filename) as f:
#         for _ in range(args.buffersize):
#             data = f.buffer_read(args.blocksize, ctype='float')
#             if not data:
#                 break
#             q.put_nowait(data)  # Pre-fill queue

#         
#         with stream:
#             timeout = args.blocksize * args.buffersize / f.samplerate
#             while data:
#                 data = f.buffer_read(args.blocksize, ctype='float')
#                 q.put(data, timeout=timeout)
#             event.wait()  # Wait until playback is finished
