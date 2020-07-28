import soundfile as sf
import sounddevice as sd
import logging
import numpy as np
import os

from . import core
from . import config

from . import effects

class Sampler(object):

    def __init__(self, init=None):
        
        self.samples = dict()
        self.load(0, 'Kick 05.wav')
        self.load(1, 'Snare 03.wav')
        self.load(2, 'Hat 01.wav')
        if init is not None:
            for ikey in init:
                self.load(ikey, init[ikey]) 
            
    def load(self, name, path):
        try:
            if not os.path.exists(path):
                path = config.SAMPLES_FOLDER + path
            with open(path, 'rb') as f:
                data, samplerate = sf.read(f)
                if len(data.shape) == 1:
                    data = np.array([data, data]).T
                if len(data.shape) != 2:
                    raise Exception('bad sample format: {}'.format(data.shape))
                if data.shape[1] != 2:
                    raise Exception('bad sample format: {}'.format(data.shape))
        except Exception as e:
            logging.warn('error reading sample {}: {}'.format(path, e))
        else:
            self.set(name, data)
            logging.info('sample {} loaded on tape {}'.format(path, name))
    
    def get_sample(self, name):
        return Sample(name, len(self.samples[name][0]))

    def next(self, sample):
        first = sample.index
        last = sample.next()
        block = list()
        for ich in range(config.CHANNELS):
            block.append(self.samples[sample.name][ich][first:last])
        return block

    def set(self, name, data):
        self.samples[name] = list()
        data = data.astype(config.DTYPE)
        if len(data)%config.BLOCKSIZE != 0:
            data = np.concatenate((data, np.zeros_like(data)[:config.BLOCKSIZE - len(data)%config.BLOCKSIZE,:]))
        for ich in range(data.shape[1]):
            self.samples[name].append(memoryview(data[:,ich]))

        
    def get(self, name):
        return np.array((self.samples[name][0][:],
                         self.samples[name][1][:])).T
    def play(self, name):
        sd.play(self.get(name),
                config.SAMPLERATE, device=config.DEVICE, blocking=False)

    def apply(self, name, effect, *args):
        data = getattr(effects, effect)(self.get(name), *args)
        self.set(name, data)

    def save(self, name, path):
        sf.write(path, self.get(name), config.SAMPLERATE)
        
class Sample(object):

    def __init__(self, name, length):
        self.index = 0
        self.name = name
        self.length = int(length)

    def isfinished(self):
        if self.index >= self.length - 1:
            return True
        return False
    
    def next(self):
        self.index += config.BLOCKSIZE
        return int(self.index)
            
    
