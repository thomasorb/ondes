import numpy as np
import time
import logging
import soundfile as sf

from . import core
from . import config
from . import sampler

class Sequencer(object):

    def __init__(self, data, sampler_init):
        assert isinstance(data, core.Data)
        self.data = data
        self.sampler = sampler.Sampler(init=sampler_init)
        
        self.absstep = 0
        self.step = 0
        self.measure = 0
        self.last_step_time = time.time()

        self.data['score0'][0] = True
        
        self.data['score1'][4] = True
        self.data['score0'][10] = True
        self.data['score1'][12] = True
        self.data['score1'][15] = True
        
        self.data['score2'][:] = np.ones_like(self.data['score2'][:])

        self.samples = list()
                
        while True:
            if not self.data.play.get():
                time.sleep(config.SLEEPTIME)
                continue
            
            now = time.time()
            step_timing = 60./self.data.tempo.get()/self.data.steps.get()*config.BEATS
            if now - self.last_step_time >= step_timing:
                self.last_step_time = now
                self.absstep += 1
                self.step += 1
                if self.step >= self.data.steps.get():
                    self.step = 0
                    self.measure += 1

                self.clean_samples()
                
                for inst in range(config.MAX_INSTRUMENTS):
                    if (self.data['score{}'.format(inst)][self.step]):
                        self.samples.append(self.sampler.get_sample(inst))
                        
            if not self.data.q.qsize() >= config.BUFFERSIZE:
                self.data.qput(self.next())
            time.sleep(config.SLEEPTIME)
                    
                
    def next(self):
        block = np.zeros((config.BLOCKSIZE, config.CHANNELS), dtype=config.DTYPE)
        for isample in self.samples:
            if not isample.isfinished():
                idata = self.sampler.next(isample)
                for ich in range(config.CHANNELS):
                    block[:,ich] += idata[ich][:]
        return block

        
    def clean_samples(self):
        clean_list = list()
        while len(self.samples) > 0:
            isample = self.samples.pop(0)
            if not isample.isfinished():
                clean_list.append(isample)
        self.samples = clean_list
                    
                    
                        

                       
