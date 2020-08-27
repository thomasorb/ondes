import soundfile as sf
import sounddevice as sd
import logging
import numpy as np
import os
import time

from . import core
from . import config

from . import effects

class Sample(object):

    def __init__(self, path):
        if isinstance(path, str):
            if not os.path.exists(path):
                path = config.SAMPLES_FOLDER + path
            with open(path, 'rb') as f:
                sample, samplerate = sf.read(f)
                sample /= config.VOLUME_ADJUST # volume adjust
                sample = sample.astype(config.DTYPE)
                if len(sample.shape) == 1:
                    sample = np.array([sample, sample]).T
                if len(sample.shape) != 2:
                    raise Exception('bad sample format: {}'.format(sample.shape))
                if sample.shape[1] != 2:
                    raise Exception('bad sample format: {}'.format(sample.shape))
            if len(sample)%2:
                sample = np.concatenate((sample, np.zeros((1, sample.shape[1]), dtype=sample.dtype)))
                
            self.sample = sample
        elif isinstance(path, np.ndarray):
            self.sample = path
        else:
            raise Exception('unknown sample')
                    
    def play(self, data=None):
        if data is None:
            sd.play(self.sample, config.SAMPLERATE, device=config.DEVICE, blocking=False)
        else:
            core.play_on_buffer('sampler', data, self.sample)                

    def apply(self, effect, *args):
        sample = getattr(effects, effect)(self.sample, *args)
        if len(sample)%2:
            sample = np.concatenate((sample, np.zeros((1, sample.shape[1]), dtype=sample.dtype)))
        self.sample = sample
            
    def save(self, path):
        sf.write(path, self.sample, config.SAMPLERATE)
        

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
            sample = self.open(path)
        except Exception as e:
            logging.warn('error reading sample {}: {}'.format(path, e))
        else:
            self.data.set_sample(name, sample)
            logging.info('sample loaded on tape {} ({} blocks)'.format(path, name, self.data.get_sample_size(name)))
            
    def open(self, path):
        sample = Sample(path)
        return sample.sample
                
    def play(self, name):
        Sample(self.data.get_sample(name)).play(data=self.data)

    def apply(self, name, effect, *args):
        sample = Sample(self.data.get_sample(name))
        sample.apply(effect, *args)
        self.data.set_sample(name, sample.sample)
        logging.info('sample set on tape {} ({} blocks)'.format(name, self.data.get_sample_size(name)))
            
    def save(self, name, path):
        Sample(self.data.get_sample(name)).save(path)
        
            
    
