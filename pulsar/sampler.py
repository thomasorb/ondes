import soundfile as sf
import sounddevice as sd
import logging
import numpy as np
import os

from . import core
from . import config

from . import effects

class Sample(object):

    def __init__(self, path):
        if isinstance(path, str):
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
            if len(data)%2:
                data = np.concatenate((data, np.zeros((1, data.shape[1]), dtype=data.dtype)))
                
            self.data = data
        elif isinstance(path, np.ndarray):
            self.data = path
        else:
            raise Exception('unknown data')
                    
    def play(self):
        sd.play(self.data, config.SAMPLERATE, device=config.DEVICE, blocking=False)

    def apply(self, effect, *args):
        data = getattr(effects, effect)(self.data, *args)
        if len(data)%2:
            data = np.concatenate((data, np.zeros((1, data.shape[1]), dtype=data.dtype)))
        self.data = data
            

    def save(self, path):
        sf.write(path, self.data, config.SAMPLERATE)
        

class Sampler(object):

    def __init__(self, data, init=None):
        assert isinstance(data, core.Data)
        self.data = data
        
        self.load(0, 'Kick 05.wav')
        self.load(1, 'Snare 03.wav')
        self.load(2, 'Hat 01.wav')
        if init is not None:
            for ikey in init:
                self.load(ikey, init[ikey]) 
            
    def load(self, name, path):
        try:
            data = self.open(path)
        except Exception as e:
            logging.warn('error reading sample {}: {}'.format(path, e))
        else:
            self.data.set_sample(name, data)
            logging.info('sample loaded on tape {} ({} blocks)'.format(path, name, self.data.get_sample_size(name)))
            
    def open(self, path):
        sample = Sample(path)
        return sample.data
                
    def play(self, name):
        Sample(self.data.get_sample(name)).play()

    def apply(self, name, effect, *args):
        sample = Sample(self.data.get_sample(name))
        sample.apply(effect, *args)
        self.data.set_sample(name, sample.data)
        logging.info('sample set on tape {} ({} blocks)'.format(name, self.data.get_sample_size(name)))
            
    def save(self, name, path):
        Sample(self.data.get_sample(name)).save(path)
        
            
    
