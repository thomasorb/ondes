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
        
class Keyboard(object):
    
    def __init__(self, data):
        
        assert isinstance(data, core.Data)
        self.data = data
                               
        # fill sample
        # eventually the synth will be changing the sample from the x, y inputs
        # the Keyboard is only seeing the sample and samples into it
        time.sleep(0.5)
            
        self.inport = mido.open_input(self.get_device())

        #last_note = None
        while True:
            stime = time.time()
            
            
            time.sleep(config.SLEEPTIME)
            
            for msg in self.inport.iter_pending():
                if msg.type == 'note_on':
                    self.data['note{}'.format(msg.note)] = True
                elif msg.type == 'note_off':
                    self.data['note{}'.format(msg.note)] = False
                elif msg.type == 'aftertouch':
                    pass
                    # for ikeys in self.keys:
                    #     if ikeys.note == last_note:
                    #         ikeys.set_volume(msg.value / 127.)
                elif msg.type == 'control_change':
                    if str(msg.control) in config.CC_IN:
                        self.data['cc{}'.format(msg.control)].set(msg.value)
                    else:
                        print(msg.control)
                        
            self.data.timing_buffers['midi_loop_time'].put(time.time() - stime)
        self.inport.close()
                
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
