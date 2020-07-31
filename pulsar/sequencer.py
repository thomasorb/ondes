import numpy as np
import time
import logging
import soundfile as sf

from . import core
from . import config
from . import sampler
from . import utils
from . import synth

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

    basename = 'track'
    def __init__(self, data, index):
        assert isinstance(data, core.Data)
        self.data = data
        self.index = int(index)
        self.name = '{}{}'.format(self.basename, self.index)
        assert hasattr(self.data, self.name), 'bad track index'

    def set(self, track):
        self.data[self.name][:len(track)] = track

    def get(self):
        return self.data[self.name][:self.data.steps.get()]
        
    def __lshift__(self, msg):
        if isinstance(msg, str):
            msg = utils.read_msg(msg, self.data.steps.get())
        elif not isinstance(msg, np.ndarray):
            raise Exception('bad msg type')
        self.set(msg)

    def __repr__(self):
        track = self.get()
        string = list(np.where(track, 'x', '.', ))
        for i in range(0, len(string), 9):
            string.insert(i, '|')
        return ''.join(string)

    __str__ = __repr__


class STrack(Track):
    basename = 'strack'

    def set(self, track, trackd):
        self.data[self.name][:len(track)] = track
        self.data[self.name + 'd'][:len(trackd)] = trackd

    def get(self):
        return (self.data[self.name][:self.data.steps.get()],
                self.data[self.name + 'd'][:self.data.steps.get()])

    def __lshift__(self, msg):
        if isinstance(msg, str):
            msg = utils.read_synth_msg(msg, self.data.steps.get())
        elif not isinstance(msg, tuple):
            raise Exception('bad msg type')
        self.set(*msg)

    
    def __repr__(self):
        track, trackd = self.get()
        string = list()
        dur = 0
        for i, d in zip(track, trackd):
            if i >= 0:
                string.append(utils.get_note_name(i))
                dur = int(d) - 1
            elif dur > 0:
                string.append(' = ')
                dur -= 1
                
            else:
                string.append(' . ')
            
        for i in range(0, len(string), 9):
            string.insert(i, '|')
        
        return ''.join(string)

    __str__ = __repr__
    
class SamplerScore(object):

    max_scores = config.MAX_INSTRUMENTS
    def __init__(self, data):
        assert isinstance(data, core.Data)
        self.data = data

    def __getitem__(self, index):
        return Track(self.data, index)

    def __repr__(self):
        string = ''
        for itrack in range(self.max_scores):
            string += '{}: {}\n'.format(itrack, self[itrack])
        return string
    
    __str__ = __repr__

class SynthScore(SamplerScore):
    
    max_scores = config.MAX_SYNTHS
    def __getitem__(self, index):
        return STrack(self.data, index)


class Sequencer(object):

    def __init__(self, data):
        assert isinstance(data, core.Data)
        self.data = data
        self.step = -1
        self.measure = -1
        
        
        self.data['track0'][0] = True
        
        self.data['track1'][8] = True
        self.data['track0'][20] = True
        self.data['track1'][24] = True
        #self.data['track1'][30] = True
        
        self.data['track2'][:32] = np.array(list([1,0]) * 16)
        self.data['track2'][31] = True
        
        self.data['strack0'][0] = 24
        self.data['strack0d'][0] = 16
        self.data['strack0'][16] = 26
        self.data['strack0d'][16] = 16
        
        self.samples = list()
                       
        # load synths
        self.synths = list()
        for isynth in range(config.MAX_SYNTHS):
            self.synths.append(synth.Synth(isynth, self.data))

        self.last_step_time = 0#time.time()
        while True:
            # if paused
            if not self.data.play.get():
                time.sleep(config.SLEEPTIME)
                self.step = -1 # restart at sequencer at the beginning of a measure
                self.samples = list()
                continue
            
            now = time.time()
            step_timing = 60. / self.data.tempo.get() / self.data.steps.get() * config.BEATS
            if now - self.last_step_time >= step_timing:
                self.last_step_time = now
                self.step += 1
                if self.step >= self.data.steps.get():
                    self.step = 0
                    self.measure += 1

                self.clean_samples()
                
                for inst in range(config.MAX_INSTRUMENTS):
                    if (self.data['track{}'.format(inst)][self.step]):
                        try:
                            self.samples.append(SampleIndex(inst, self.data.get_sample_size(inst)))
                        except AttributeError:
                            logging.warn('sample {} not loaded'.format(inst))

                for isynth in range(config.MAX_SYNTHS):
                    if (self.data['strack{}'.format(isynth)][self.step ]>= 0):
                        iindex = config.MAX_INSTRUMENTS + isynth
                        inote = self.data['strack{}'.format(isynth)][self.step]
                        idur = self.data['strack{}d'.format(isynth)][self.step]
                        isynth_sample = self.synths[isynth].get_samples(inote, idur)
                        self.data.set_sample(iindex, isynth_sample)
                        try:
                            self.samples.append(SampleIndex(
                                iindex, len(isynth_sample)))
                        except AttributeError:
                            logging.warn('synth {} get sample error'.format(inst))
                            
            if not self.data.buffer_is_full('sequencer'):
                try:
                    self.data.put_block('sequencer', *self.next_block())
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
                    
                    
                        

                       
