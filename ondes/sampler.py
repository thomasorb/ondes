import soundfile as sf
import sounddevice as sd
import logging
import numpy as np
import os
import time
import traceback

from . import core
from . import config

from . import effects

class Effect(object):
    
    def __init__(self, name, data):
        self.data = data
        self.effect = getattr(effects, name)
        
    def __call__(self, *args):
        return Sample(self.effect(self.data, *args))


class Sample(object):

    def __init__(self, path):
        if isinstance(path, str):
            if not os.path.exists(path):
                path = config.SAMPLES_FOLDER + path
            with open(path, 'rb') as f:
                sample, samplerate = sf.read(f)
                #sample /= config.VOLUME_ADJUST # volume adjust
                sample = sample.astype(config.DTYPE)
                if len(sample.shape) == 1:
                    sample = np.array([sample, sample]).T
                if len(sample.shape) != 2:
                    raise Exception('bad sample format: {}'.format(sample.shape))
                if sample.shape[1] != 2:
                    raise Exception('bad sample format: {}'.format(sample.shape))

            
            #if len(sample)%2:
            #    sample = np.concatenate((sample, np.zeros((1, sample.shape[1]), dtype=sample.dtype)))
            sample = effects.cut_to_blocksize(sample, config.BLOCKSIZE)
                
            self.sample = sample
        elif isinstance(path, np.ndarray):
            if len(path.shape) == 1:
                path = np.array((path, path)).T
            sample = effects.cut_to_blocksize(path, config.BLOCKSIZE)
            
            self.sample = sample
        else:
            raise Exception('unknown sample')


    def __len__(self):
        return len(self.sample)
    
    def play(self, data=None, duration=None):
        sample = self.sample
        if duration is not None:
            new_size = int(duration * config.SAMPLERATE)
            if len(sample) > new_size:
                sample = np.copy(sample)
                sample = sample[:new_size, :]
                sample = effects.cut_to_blocksize(sample, config.BLOCKSIZE)
                
        if data is None:
            import multiprocessing as mp
            def sd_play_sample(sample, srate, device):
                try:
                    sd.play(sample, srate, device=device, 
                            never_drop_input=False, blocksize=config.BLOCKSIZE, blocking=True)
                    sd.wait()
                except:
                    print('error when playing sample')
                    print(sd.query_devices())
                    
                    traceback.print_exc(5)
                    
            proc = mp.Process(name='play', target=sd_play_sample, args=(
                sample, config.SAMPLERATE, config.DEVICE))
            proc.start()
            
        else:
            core.play_on_buffer('sampler', data, sample)

    def apply(self, effect, *args):
        sample = getattr(effects, effect)(self.sample, *args)
        sample = effects.cut_to_blocksize(sample, config.BLOCKSIZE)
        self.sample = sample
            
    def save(self, path):
        sf.write(path, self.sample, config.SAMPLERATE)

    def copy(self):
        return self.__class__(np.copy(self.sample))

    def __getattr__(self, effect):
        return Effect(effect, self.sample)
        

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
        
            
    
