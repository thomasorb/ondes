import multiprocessing as mp
import logging

from . import sequencer
from . import core
from . import player
from . import sampler

class Machine(object):

    def __init__(self):
        

        logging.getLogger().setLevel(logging.DEBUG)
        self.data = core.Data()
        self.processes = list()

        #self.add_process('sampler', sampler.Sampler, (self.data, sampler_init))
        self.add_process('sequencer', sequencer.Sequencer, (self.data,))
        self.add_process('player', player.Player, (self.data,))
        
        for iproc in self.processes:
            iproc.start()

        

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
