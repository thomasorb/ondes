import numpy as np
import logging
import sounddevice as sd
import scipy.interpolate
import time

from . import core
from . import ccore
from . import utils
from . import config


class Wave(object):

    def __init__(self, mode='sine', dirty=300):
        self.wave = ccore.Wave(mode=str(mode), dirty=int(dirty))

        # get base sample
        base = np.array(self.wave.get_base_samples()).T

        # downsample it to the 0 note
        sampleL = utils.inverse_transform(base[:,0], 0, config.BASENOTE, config.A_MIDIKEY).real
        sampleR = utils.inverse_transform(base[:,1], 0, config.BASENOTE, config.A_MIDIKEY).real
        
        sampleR = sampleR[ccore.get_first_zero(sampleL):ccore.get_last_zero(sampleL)]
        sampleL = sampleL[ccore.get_first_zero(sampleL):ccore.get_last_zero(sampleL)]
        self.downsample = np.array((sampleL, sampleR)).T
        
class Synth(object):

    def __init__(self, name, data, mode='sine', dirty=300):
        
        assert isinstance(data, core.Data)
        self.data = data
        self.name = 's{}'.format(name)
        
        wave = Wave(mode=mode, dirty=dirty)
        self.downsample = np.copy(wave.downsample)
        self.downsamplef = scipy.interpolate.interp1d(
            np.arange(len(self.downsample)),
            self.downsample, axis=0)
        
        self.data.set_sample(self.name, np.copy(self.downsample))
        self._hash = self.data[self.name + 'hash'][:]
        
    def get_samples(self, note, duration):
        """
        :param duration: in steps
        """        
        # upsampling

        # change wave data if it has changed on disk
        if self._hash != self.data[self.name + 'hash'][:]:
            self._hash = self.data[self.name + 'hash'][:]
            
            self.downsample = self.data.get_sample(self.name)
            self.downsamplef = scipy.interpolate.interp1d(
                np.arange(len(self.downsample)),
                self.downsample, axis=0)

        ratio = utils.note2f(0, config.A_MIDIKEY) / utils.note2f(note, config.A_MIDIKEY)
        
        sample = self.downsamplef(
            np.linspace(0, len(self.downsample) - 1, int(ratio * len(self.downsample))))

        # step timing in s
        step_timing = 60. / self.data.tempo.get() / self.data.steps.get() * config.BEATS
        
        sample_duration = len(sample) / config.SAMPLERATE # in s
        sample_duration /= step_timing # in steps
        final_size = int(len(sample) * duration / sample_duration)
        if final_size > len(sample):
            ratio = (final_size // len(sample) + 1)
            sample = np.concatenate(list([sample,]) * ratio)

        if final_size <= len(sample):
            sample = sample[:final_size]
            
        return sample


    def play(self, note, duration):
        sample = self.get_samples(note, duration)

        index = 0
        while index < len(sample) - 1:
            if self.data.buffer_is_full('synth'):
                time.sleep(config.SLEEPTIME)
                continue

            self.data.put_block(
                'synth',
                sample[index:index+config.BLOCKSIZE,0],
                sample[index:index+config.BLOCKSIZE,1])
            index += config.BLOCKSIZE
            
        #def add(self, wave)

