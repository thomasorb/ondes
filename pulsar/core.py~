import numpy as np
import ctypes
import multiprocessing.sharedctypes
import multiprocessing as mp
import logging
from . import config

class Value(object):

    def __init__(self, init):
        if isinstance(init, float):
            ctype = ctypes.c_double
        elif isinstance(init, int):
            ctype = ctypes.c_int
            
        else: raise TypeError('unrecognized type: {}'.format(type(init)))
            
        self.val = multiprocessing.sharedctypes.Value(ctype, init, lock=True)            
        
        
    def set(self, value):
        self.val.value = value

    def get(self):
        return self.val.value

class Array(object):

    def __init__(self, init):
        assert isinstance(init, np.ndarray), 'init must be a numpy.ndarray'
        if init.dtype == np.uint16:
            ctype = ctypes.c_uint16
        elif init.dtype == np.bool:
            ctype = ctypes.c_bool
        
        else: raise TypeError('unrecognized type: {}'.format(init.dtype))
                    
        self.val = multiprocessing.sharedctypes.Array(ctype, init.size, lock=True)
        self.val[:] = init

    def __setitem__(self, *args):
        self.val.__setitem__(*args)

    def __getitem__(self, *args):
        return self.val.__getitem__(*args)
    
class Data(object):

    def __init__(self):

        self.add_value('play', False)
        self.add_value('tempo', config.TEMPO)
        self.add_value('steps', config.STEPS)
        for i in range(config.MAX_INSTRUMENTS):
            self.add_array('score{}'.format(i), np.zeros(config.MAX_STEPS_PER_MEASURE,
                                                         dtype=np.bool))
        self.q = mp.Queue(maxsize=config.BUFFERSIZE)
        self.q_lock = mp.Lock()

        self.event = mp.Event()
        
    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)
    
    def add_value(self, name, init):
        try:
            getattr(self, name)
        except AttributeError:
            setattr(self, name, Value(init))
        else: logging.warn('attribute {} already exists'.format(name))

    def add_array(self, name, init):
        try:
            getattr(self, name)
        except AttributeError:
            setattr(self, name, Array(init))
        else: logging.warn('attribute {} already exists'.format(name))

    def qput(self, *args):
        self.q_lock.acquire()
        self.q.put_nowait(*args)
        self.q_lock.release()

    def qget(self):
        with self.q_lock:
            return self.q.get_nowait()
        
