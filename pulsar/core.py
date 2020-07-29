import numpy as np
import ctypes
import multiprocessing.sharedctypes
import multiprocessing as mp
import logging
from . import config

class BufferEmpty(Exception): pass

class BufferFull(Exception): pass

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
        if init.dtype == np.float32:
            ctype = ctypes.c_float
        elif init.dtype == np.uint16:
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

    def __len__(self):
        return len(self.val)
    
class Data(object):

    def __init__(self):

        self.add_value('play', False)
        self.add_value('tempo', config.TEMPO)
        self.add_value('steps', config.STEPS)
        for i in range(config.MAX_INSTRUMENTS):
            self.add_array('score{}'.format(i), np.zeros(config.MAX_STEPS_PER_MEASURE,
                                                         dtype=np.bool))
        self.buffer_lock = mp.Lock()
        self.add_array('bufferL', np.zeros(config.BLOCKSIZE * config.BUFFERSIZE, dtype=config.DTYPE))
        self.add_array('bufferR', np.zeros(config.BLOCKSIZE * config.BUFFERSIZE, dtype=config.DTYPE))
        self.add_value('next_read_block', 0)
        self.add_value('next_write_block', 0)
        self.add_value('buffer_counts', 0)

        for isample in range(config.MAX_SAMPLES):
            self.set_sample(
                isample, np.zeros(
                    (config.BLOCKSIZE * config.MAX_SAMPLE_LEN, 2),
                    dtype=config.DTYPE),
                config.BLOCKSIZE*4)


    def buffer_is_full(self):
        if self.buffer_counts.get() >= config.BUFFERSIZE:
            return True
        else: return False
            
    def put_block(self, blockL, blockR):
        with self.buffer_lock:    
            if self.buffer_counts.get() >= config.BUFFERSIZE:
                raise BufferFull
            
            next_write = self.next_write_block.get()
            
            self.bufferL[next_write * config.BLOCKSIZE:
                         (next_write + 1) * config.BLOCKSIZE] = blockL
            self.bufferR[next_write * config.BLOCKSIZE:
                         (next_write + 1) * config.BLOCKSIZE] = blockR
            next_write += 1
            if next_write >= config.BUFFERSIZE:
                next_write = 0
            self.next_write_block.set(next_write)
            self.buffer_counts.set(self.buffer_counts.get() + 1)
            
        
    def get_block(self):
        with self.buffer_lock:
            next_read = self.next_read_block.get()
            
            if self.buffer_counts.get() <= 0:
                raise BufferEmpty
            buf = np.array((self.bufferL[next_read * config.BLOCKSIZE:
                                         (next_read + 1) * config.BLOCKSIZE],
                            self.bufferR[next_read * config.BLOCKSIZE:
                                         (next_read + 1) * config.BLOCKSIZE])).T
            next_read += 1
            if next_read >= config.BUFFERSIZE:
                next_read = 0
                
            self.next_read_block.set(next_read)
            self.buffer_counts.set(self.buffer_counts.get() - 1)
            return buf

    def set_sample(self, name, data, length=None):
        data = data.astype(config.DTYPE)
        if len(data)%config.BLOCKSIZE != 0:
            data = np.concatenate(
                (data, np.zeros_like(data)[:config.BLOCKSIZE - len(data)%config.BLOCKSIZE,:]))

        if length is None:
            length = len(data) # not necessary the true len, useful at init
        size = min(len(data), config.BLOCKSIZE * config.MAX_SAMPLE_LEN)
        try:
            getattr(self, str(name) + 'L')
        except:            
            self.add_array(str(name) + 'L', data[:size,0])
            self.add_array(str(name) + 'R', data[:size,1])    
            self.add_value(str(name) + 'len', length)
        else:
            self[str(name) + 'L'][:size] = data[:size,0]
            self[str(name) + 'R'][:size] = data[:size,1]
            self[str(name) + 'len'].set(length)    

    def get_sample_size(self, name):
        """Return size in buffers"""
        return int(self[str(name) + 'len'].get() // config.BLOCKSIZE)
    
    def get_sample_block(self, name, buffer_index):
        block = list()
        if buffer_index > self.get_sample_size(name):
            raise Exception('sample finished')
        block.append(self[str(name) + 'L'][buffer_index * config.BLOCKSIZE:
                                           (buffer_index + 1) * config.BLOCKSIZE])
        block.append(self[str(name) + 'R'][buffer_index * config.BLOCKSIZE:
                                           (buffer_index + 1) * config.BLOCKSIZE])
        return block


    def get_sample(self, name):
        return np.array((self[str(name) + 'L'][:self[str(name) + 'len'].get()],
                         self[str(name) + 'R'][:self[str(name) + 'len'].get()])).T

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
        else:
            logging.warn('attribute {} already exists'.format(name))

