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

        devices = mido.get_output_names()
        logging.info('midi devices:\n{}'.format('\n  '.join(devices)))
        self.controllers = list()
        registered_ccs = dict()
        
        for icontrol in range(len(config.MIDIDEVICES)):
            idev = mido.open_input(self.get_device(icontrol))
            if idev is not None:
                self.controllers.append(idev)
                registered_ccs[icontrol] = dict()

        for icc in config.CC_MATRIX:
            registered_ccs[config.CC_MATRIX[icc][0]][config.CC_MATRIX[icc][1]] = icc

        
        while True:
            stime = time.time()
                        
            time.sleep(config.SLEEPTIME)

            for icontrol in range(len(self.controllers)):
                for msg in self.controllers[icontrol].iter_pending():
                    if msg.type == 'note_on':
                        if msg.velocity > 0:
                            self.data['note{}'.format(msg.note)].set(True)
                            self.data['vel{}'.format(msg.note)].set(msg.velocity)
                        else: # interpreted as note off
                            self.data['note{}'.format(msg.note)].set(False)
                        
                    elif msg.type == 'note_off':
                        self.data['note{}'.format(msg.note)].set(False)
                    elif msg.type == 'aftertouch':
                        pass
                        # for ikeys in self.keys:
                        #     if ikeys.note == last_note:
                        #         ikeys.set_volume(msg.value / 127.)
                    elif msg.type == 'control_change':
                        if msg.control in registered_ccs[icontrol]:
                            self.data['cc_{}'.format(registered_ccs[icontrol][msg.control])].set(msg.value)
                        elif not msg.value:
                            logging.info('unregisterd input {}'.format(msg.control))
                        
            self.data.timing_buffers['midi_loop_time'].put(time.time() - stime)
        for icontrol in self.controllers:
            self.controllers[icontrol].close()
                
    def get_device(self, icontrol):
        devices = mido.get_output_names()
        for device in devices:
            if config.MIDIDEVICES[icontrol] in device:
                return device

        logging.warning('device not found: {}'.format(config.MIDIDEVICES[icontrol]))
        return None

    def __del__(self):
        for icontrol in self.controllers:
            try:
                self.controllers[icontrol].close()
            except: pass
