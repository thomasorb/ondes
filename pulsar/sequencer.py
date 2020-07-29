import numpy as np
import time
import logging
import soundfile as sf

from . import core
from . import config
from . import sampler
from . import utils

class AllFinished(Exception): pass

class SampleIndex(object):

    def __init__(self, name, length):
        self.index = 0
        self.name = name
        self.length = int(length)

    def isfinished(self):
        if self.index >= self.length - 1:
            return True
        return False
    
    def next(self):
        this_index = int(self.index)
        self.index += 1
        return this_index

class Track(object):

    def __init__(self, data, index):
        assert isinstance(data, core.Data)
        self.data = data
        self.index = int(index)
        assert hasattr(self.data, 'score{}'.format(index)), 'bad track index'

    def __lshift__(self, msg):
        if isinstance(msg, str):
            msg = utils.read_msg(msg, self.data.steps.get())
        elif not isinstance(msg, np.ndarray):
            raise Exception('bad msg type')
        self.data['score{}'.format(self.index)][:len(msg)] = msg

    def __repr__(self):
        track = self.data['score{}'.format(self.index)][:self.data.steps.get()]
        string = list(np.where(track, 'x', '.', ))
        for i in range(0, len(string), 9):
            string.insert(i, '|')
        return ''.join(string)

    __str__ = __repr__
    
class Score(object):

    def __init__(self, data):
        assert isinstance(data, core.Data)
        self.data = data

    def __getitem__(self, index):
        return Track(self.data, index)

    def __repr__(self):
        string = ''
        for itrack in range(config.MAX_INSTRUMENTS):
            string += '{}: {}\n'.format(itrack, self[itrack])
        return string
    __str__ = __repr__


class Sequencer(object):

    def __init__(self, data):
        assert isinstance(data, core.Data)
        self.data = data
        self.absstep = 0
        self.step = 0
        self.measure = 0
        self.last_step_time = time.time()

        
        self.data['score0'][0] = True
        
        self.data['score1'][8] = True
        self.data['score0'][20] = True
        self.data['score1'][24] = True
        #self.data['score1'][30] = True
        
        self.data['score2'][:32] = np.array(list([1,0]) * 16)
        self.data['score2'][31] = True
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
                        try:
                            self.samples.append(SampleIndex(inst, self.data.get_sample_size(inst)))
                        except AttributeError:
                            logging.warn('sample {} not loaded'.format(inst))
                            
            if not self.data.buffer_is_full():
                try:
                    self.data.put_block(*self.next_block())
                except AllFinished: pass
                except Exception as e:
                    logging.warn('error at put block {}'.format(e))
                    raise
            time.sleep(config.SLEEPTIME)
                    
                
    def next_block(self):
        blockL = np.zeros(config.BLOCKSIZE, dtype=config.DTYPE)
        blockR = np.zeros(config.BLOCKSIZE, dtype=config.DTYPE)
        all_finished = True
        for isample in self.samples:
            if not isample.isfinished():
                try:
                    iblock = self.data.get_sample_block(
                        isample.name, isample.index)
                except Exception:
                    pass 
                else:
                    if len(iblock[0][:]) == config.BLOCKSIZE:
                        blockL += iblock[0][:]
                        blockR += iblock[1][:]
                        isample.next()   
                        all_finished = False
        if all_finished:
            raise AllFinished
        return blockL, blockR
    
    def clean_samples(self):
        clean_list = list()
        while len(self.samples) > 0:
            isample = self.samples.pop(0)
            if not isample.isfinished():
                clean_list.append(isample)
        self.samples = clean_list
                    
                    
                        

                       
