import mido
import logging
import numpy as np
import time
import scipy.interpolate
import scipy.stats

from . import ccore

from . import core
from . import config
from . import synth
from . import utils

class AllFinished(Exception): pass

class KeySampler(object):
    def __init__(self, data, note, velocity, length, attack=10, release=10, velocity_response=3):
        """
        :param attack: in ms
        :param release: in ms
        """
        assert isinstance(data, core.Data)
        self.data = data

        self.index = 0
        self.note = int(note)
        self.volume = int(velocity) / 127.
        velocity = int(velocity) / 127. * float(velocity_response)
        self.attack = 5 + float(attack) / velocity 
        self.release = float(release) * velocity * 30
        self.released = False
        self.stopped = False
        self.length = int(length)
        self.sampling_step = utils.note2f(
            self.note + config.PADNOTE_SHIFT, config.A_MIDIKEY) / utils.note2f(0, config.A_MIDIKEY)
        self.duration = 0 # time since start in ms
        self.t = np.arange(config.BLOCKSIZE) / config.SAMPLERATE * 1000 # in ms

    def stop(self):
        self.released = True
        self.release_start = float(self.duration)

    def set_volume(self, volume):
        self.volume = float(volume)

        
    def get_next(self):
        start_index = int(self.index)
        
        shift = 10**((self.data[config.CC_SHIFT].get() / 127. - 0.5)*2)
        
        end_index = start_index + config.BLOCKSIZE * self.sampling_step * shift
        sampling_vector = np.linspace(start_index, end_index, config.BLOCKSIZE)
        sampling_vector = np.mod(sampling_vector, self.length - 1)
        self.index = sampling_vector[-1] + self.sampling_step

        envelope = np.clip((self.t + self.duration) / self.attack, 0, 1)

        if self.released:
            envelope *= np.clip(-(self.t + self.duration - self.release_start)/self.release + 1, 0, 1)
        self.duration += config.BLOCKTIME
        if self.released:
            if self.duration - self.release_start > self.release:
                self.stopped = True
        
        return sampling_vector.astype(config.DTYPE), envelope * self.volume


class SynthData(object):

    def __init__(self, data, index):
        assert isinstance(data, core.Data)
        self.data = data

        self.index = int(index)
        self.name = 's{}'.format(index)

        self.length = self.data.samples[self.name].get_len()
        self.sample_hash = self.data.samples[self.name].get_hash()
        self.sample = self.data.samples[self.name].get_sample().astype(config.DTYPE)
        self.old_sample = np.copy(self.sample)
        self.transit_start = time.time()
        self.samples_counter = 0

    def update(self):
        new_hash = self.data.samples[self.name].get_hash()
        if new_hash != self.sample_hash:
            self.sample_hash = self.data.samples[self.name].get_hash()
            self.old_sample = np.copy(self.sample)
            self.sample = self.data.samples[self.name].get_sample().astype(config.DTYPE)
            self.transit_start = int(self.samples_counter)
            return True
        else:
            return False
            
        
        
class Keyboard(object):
    
    def __init__(self, data):
        
        assert isinstance(data, core.Data)
        self.data = data
                               
        # fill sample
        # eventually the synth will be changing the sample from the x, y inputs
        # the Keyboard is only seeing the sample and samples into it
        time.sleep(0.5)

        self.synths = list()
        for i in range(config.MAX_SYNTHS):
            self.synths.append(SynthData(data, i))
            
        self.keys = list()

        self.inport = mido.open_input(self.get_device())

        last_note = None
        while True:
            stime = time.time()
            
            new_hash = self.data.samples['s0'].get_hash()

            updating = False
            for i in range(config.MAX_SYNTHS):
                if self.synths[i].update():
                    updating = True
                    
            if not updating:
                time.sleep(config.SLEEPTIME)
                self.clean_keys()
            
            for msg in self.inport.iter_pending():
                if msg.type == 'note_on':
                    self.keys.append(KeySampler(self.data, msg.note, msg.velocity,
                                                self.synths[0].length))
                    last_note = msg.note
                    
                elif msg.type == 'note_off':
                    for ikeys in self.keys:
                        if ikeys.note == msg.note:
                            ikeys.stop()
                elif msg.type == 'aftertouch':
                    pass
                    # for ikeys in self.keys:
                    #     if ikeys.note == last_note:
                    #         ikeys.set_volume(msg.value / 127.)
                elif msg.type == 'control_change':
                    if msg.control in [16, 17, 18, 19, 20, 21]:
                        self.data['cc{}'.format(msg.control)].set(msg.value)
                
            if not self.data.buffer_is_full('synth'):
                try:
                    self.data.put_block('synth', *self.next_block())
                    
                except AllFinished: pass
                except Exception as e:
                    logging.warn('error at put block {}'.format(e))
                    pass
                finally:
                    for i in range(config.MAX_SYNTHS):
                        self.synths[i].samples_counter += config.BLOCKSIZE
                        
            self.data.timing_buffers['keyboard_loop_time'].put(time.time() - stime)
        self.inport.close()
                
    def next_block(self):

        MIX = 0.25 # max 0.5
        
        stime = time.time()
        blockL = np.zeros(config.BLOCKSIZE, dtype=config.DTYPE)
        blockR = np.zeros(config.BLOCKSIZE, dtype=config.DTYPE)
            
        all_finished = True

        new = np.empty_like(blockL, dtype=complex)
        old = np.empty_like(blockL, dtype=complex)
                
        for ikey in self.keys:
            if not ikey.stopped:
                all_finished = False
                sfunc, env = ikey.get_next()
                for i in range(config.MAX_SYNTHS):
                    
                    trans = np.clip((np.arange(config.BLOCKSIZE)
                        + self.synths[i].samples_counter
                        - self.synths[i].transit_start) / config.TRANSIT_TIME, 0, 1)

        
                    oldL = ccore.fast_interp1d(np.copy(self.synths[i].old_sample[:,0]), sfunc.copy())
                    newL = ccore.fast_interp1d(np.copy(self.synths[i].sample[:,0]), sfunc.copy()) 
                    oldR = ccore.fast_interp1d(np.copy(self.synths[i].old_sample[:,1]), sfunc.copy())
                    newR = ccore.fast_interp1d(np.copy(self.synths[i].sample[:,1]), sfunc.copy())

                    # morphing
                    new.real = newL * env
                    new.imag = newR * env
                    old.real = oldL * env
                    old.imag = oldR * env

                    block = utils.morph(new, old, trans)

                    # mixing L and R
                    volume = utils.cc_rescale(self.data[getattr(config, 'CC_V{}'.format(i))].get(), 0, 1)
                    blockL += ((1 - MIX) * block.real + MIX * block.imag) * volume
                    blockR += ((1 - MIX) * block.imag + MIX * block.real) * volume

        bits = int(utils.cc_rescale(self.data[config.CC_BITS].get(), 6, 32))
        binning = int(utils.cc_rescale(self.data[config.CC_BITS].get(), 1, 32, invert=True))
        blockL = utils.bit_crush(blockL, bits, binning)
        blockR = utils.bit_crush(blockR, bits, binning)
        
                                
        if all_finished:
            raise AllFinished
        return blockL, blockR
    
    def clean_keys(self):
        clean_list = list()
        while len(self.keys) > 0:
            ikey = self.keys.pop(0)
            if not ikey.stopped:
                clean_list.append(ikey)
        self.keys = clean_list

    def get_device(self):
        devices = mido.get_output_names()
        logging.info('midi devices:\n{}'.format('\n  '.join(devices)))
        for device in devices:
            if config.MIDIDEVICE in device:
                return device

        logging.warning('device not found switching to default')
        return 'default'

    def __del__(self):
        try:
            self.inport.close()
        except: pass
