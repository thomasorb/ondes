import multiprocessing as mp
import logging

from . import core
from . import server
from . import midi
from . import synth
from . import display
from . import config

class Machine(object):

    def __init__(self, filepath, srate):
        
        logging.info('Shared memory init')
        self.data = core.Data()

        logging.info('processes init')
        self.processes = list()
        self.add_process('server', server.Server, (self.data, filepath, srate))
        
        self.add_process('midi', midi.Keyboard, (self.data,))
        #self.add_process('display', display.CubeDisplay, (self.data, dfpath))
        

        logging.info('starting processes')
        for iproc in self.processes:
            iproc.start()

        try:
            for iproc in self.processes:
                iproc.join()
        except KeyboardInterrupt:
            logging.info('Keyboard interrupt: ending processes')
            for iproc in self.processes:
                iproc.terminate()            
        

    def add_process(self, name, target, args):

        self.processes.append(
            mp.Process(name=name, target=target, args=args))

    def play(self):
        self.data.play.set(True)

    def pause(self):
        self.data.play.set(False)
        
    def __del__(self):
        try:
            for iproc in self.processes:
                iproc.terminate()
                iproc.join()
        
        except Exception as e:
            pass

