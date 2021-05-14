import numpy as np
import ctypes
import multiprocessing.sharedctypes
import multiprocessing as mp
import logging
import time

from . import config
from . import utils
from . import ccore

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
        self.is_complex = False

        if init.dtype == np.float32:
            ctype = ctypes.c_float
        elif init.dtype == np.uint16:
            ctype = ctypes.c_uint16
        elif init.dtype == np.bool:
            ctype = ctypes.c_bool
        elif init.dtype == np.int32:
            ctype = ctypes.c_long
        elif init.dtype == np.complex64:
            ctype = ctypes.c_float
            self.is_complex = True
            
        else: raise TypeError('unrecognized type: {}'.format(init.dtype))

        if not self.is_complex:
            self.val = multiprocessing.sharedctypes.Array(ctype, len(init), lock=True)
            self.val[:] = init
        
        else:
            self.val = (
                multiprocessing.sharedctypes.Array(ctype, init.size, lock=True),
                multiprocessing.sharedctypes.Array(ctype, init.size, lock=True))
            self.val[0][:] = init.real
            self.val[1][:] = init.imag
           
        
    def __setitem__(self, item, value):
        if not self.is_complex:
            self.val.__setitem__(item, value)
        else:
            self.val[0].__setitem__(item, value.real)
            self.val[1].__setitem__(item, value.imag)
        

    def __getitem__(self, item):
        #if not self.is_complex:
        return self.val.__getitem__(item)
        #else:
        #    raise NotImplementedError()
            # ret = self.val[0].__getitem__(item).astype(config.COMPLEX_DTYPE)
            # ret.imag = self.val[1].__getitem__(item)
            # return ret
        
    def __len__(self):
        return len(self.val)



class TimingBuffer(object):

    def __init__(self, data, name, length=50):
        assert isinstance(data, Data)
        self.data = data
        self.name = name
        self.length = int(length)
        
        self.buffer_lock = mp.Lock()
        self.data.add_array(self.name + 'buffer', np.full(length, np.nan, dtype=np.float32))
        self.data.add_value(self.name + 'index', 0)

    def put(self, val):
        with self.buffer_lock:
            index = int(self.data[self.name + 'index'].get())
            self.data[self.name + 'buffer'][index] = val
            index += 1
            if index >= self.length - 1:
                index = 0
            self.data[self.name + 'index'].set(index)
            
    def get(self):
        with self.buffer_lock:
            return np.array(self.data[self.name + 'buffer'][:])
        
        

class SampleBuffer(object):

    def __init__(self, data, name):
        assert isinstance(data, Data)
        self.data = data
        self.name = name

        self.buffer_lock = mp.Lock()
        self.data.add_array('bufferL' + self.name, np.zeros(
            config.BLOCKSIZE * config.BUFFERSIZE, dtype=config.DTYPE))
        self.data.add_array('bufferR' + self.name, np.zeros(
            config.BLOCKSIZE * config.BUFFERSIZE, dtype=config.DTYPE))
        self.data.add_value('next_read_block' + self.name, 0)
        self.data.add_value('next_write_block' + self.name, 0)
        self.data.add_value('buffer_counts' + self.name, 0)

        self.bufferL = self.data['bufferL' + self.name]
        self.bufferR = self.data['bufferR' + self.name]
        self.next_write_block = self.data['next_write_block' + self.name]
        self.next_read_block = self.data['next_read_block' + self.name]
        self.buffer_counts = self.data['buffer_counts' + self.name]
        
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
    
class Sample(object):

    def __init__(self, data, name, is_complex=False):
        
        assert isinstance(data, Data)
        self.data = data
        self.name = name
        self.is_complex = bool(is_complex)

        self.sample_lock = mp.Lock()

        if self.is_complex:
            self.dtype = config.COMPLEX_DTYPE
        else:
            self.dtype = config.DTYPE
            
        data = np.zeros(
            (config.BLOCKSIZE * config.MAX_SAMPLE_LEN, 2),
            dtype=self.dtype)

        self.length = len(data)

        self.data.add_value(str(self.name) + 'ready', 0)
        
        self.data.add_array(str(self.name) + '0.L', data[:,0])
        self.data.add_array(str(self.name) + '0.R', data[:,1])    
        self.data.add_value(str(self.name) + '0.len', self.length)
        self.data.add_array(str(self.name) + '0.hash', utils.get_hash())

        self.data.add_array(str(self.name) + '1.L', data[:,0])
        self.data.add_array(str(self.name) + '1.R', data[:,1])    
        self.data.add_value(str(self.name) + '1.len', self.length)
        self.data.add_array(str(self.name) + '1.hash', utils.get_hash())
        

    def get_ready(self):
        return str(self.name) + '{}.'.format(self.data[str(self.name) + 'ready'].get())
    
    def get_waiting(self):
        if self.data[str(self.name) + 'ready'].get() == 0:
            waiting = 1
        else:
            waiting = 0
        return str(self.name) + '{}.'.format(waiting)
    
    def switch(self):
        with self.sample_lock:
            _new = int(not self.data[str(self.name) + 'ready'].get())
            self.data[str(self.name) + 'ready'].set(_new)
        
    def put_sample(self, data):
        if self.is_complex:
            if np.iscomplexobj(data):
                data = data.astype(self.dtype)
            else:
                data = data.real.astype(self.dtype)
        else:
            if np.iscomplexobj(data):
                data = data.real.astype(self.dtype)
                logging.warning('sample is complex but buffer is real. Only the real part of the sample was kept')
            else:
                data = data.astype(self.dtype)

        if len(data) > config.BLOCKSIZE * config.MAX_SAMPLE_LEN:
            logging.warning('sample cut because length is larger than {}'.format(config.BLOCKSIZE * config.MAX_SAMPLE_LEN))
            data = data[:config.BLOCKSIZE * config.MAX_SAMPLE_LEN,:]

        length = len(data)

        self.data[self.get_waiting() + 'L'][:length] = data[:,0]
        self.data[self.get_waiting() + 'R'][:length] = data[:,1]
        self.data[self.get_waiting() + 'len'].set(length)
        self.data[self.get_waiting() + 'hash'][:] = utils.get_hash()
        self.switch()

    def get_sample(self):
        with self.sample_lock:
        
            # cannot use self.length because it is not updated
            length = self.get_len()
            return np.array((self.data[self.get_ready() + 'L'][:length],
                             self.data[self.get_ready() + 'R'][:length])).astype(config.DTYPE).T
            
    def get_len(self):
        return self.data[self.get_ready() + 'len'].get()

    def get_hash(self):
        return self.data[self.get_ready() + 'hash'][:]
        
        
class Data(object):

    def __init__(self):

        self.add_value('play', False)
        self.buffers = dict()
        self.timing_buffers = dict()
        self.samples = dict()

        # add buffers
        self.add_buffer('input')
        self.add_buffer('synth')
        
        # downsampled function for each synth (same stuff as a sample)
        for isynth in range(config.MAX_SYNTHS):
            self.add_sample('s{}'.format(isynth))
            # x y
            self.add_value('x{}'.format(isynth), 0)
            self.add_value('y{}'.format(isynth), 0)
            self.add_value('x_orig{}'.format(isynth), 0)
            self.add_value('y_orig{}'.format(isynth), 0)
            self.add_timing_buffer('synth_computation_time{}'.format(isynth), 30)


            # display arrays
            self.add_array('display_spectrum{}'.format(isynth), np.arange(config.MAX_DISPLAY_SIZE, dtype=config.DTYPE))
            self.add_value('display_spectrum_len{}'.format(isynth), config.MAX_DISPLAY_SIZE)
            self.add_array('display_sample{}'.format(isynth), np.arange(config.MAX_DISPLAY_SIZE, dtype=config.DTYPE))
            self.add_value('display_sample_len{}'.format(isynth), config.MAX_DISPLAY_SIZE)
        
        # timing logs
        self.add_timing_buffer('keyboard_loop_time', 100)
        self.add_timing_buffer('server_callback_time', 100)
        
        
        ## control change
        self.add_value('cc16', config.CC16_DEFAULT)
        self.add_value('cc17', config.CC17_DEFAULT)
        self.add_value('cc18', config.CC18_DEFAULT)
        self.add_value('cc19', config.CC19_DEFAULT)
        self.add_value('cc20', config.CC20_DEFAULT)
        self.add_value('cc21', config.CC21_DEFAULT)
        
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

    def add_buffer(self, name):
        self.buffers[name] = SampleBuffer(self, name)

    def add_timing_buffer(self, name, length):
        self.timing_buffers[name] = TimingBuffer(self, name, length=length)

    def buffer_is_full(self, name):
        return self.buffers[name].buffer_is_full()
    
    def get_block(self, name):
        return self.buffers[name].get_block()

    def put_block(self, name, blockL, blockR):
        return self.buffers[name].put_block(blockL, blockR)

    def add_sample(self, name, is_complex=False):
        self.samples[name] = Sample(self, name, is_complex=is_complex)
        
    def set_sample(self, name, data):
        self.samples[name].put_sample(data)
    
    def get_sample_size(self, name):
        """Return size in buffers"""
        return int(self[str(name) + 'len'].get() // config.BLOCKSIZE)
    
    # def get_sample_block(self, name, buffer_index):
    #     block = list()
    #     if buffer_index > self.get_sample_size(name):
    #         raise Exception('sample finished')
    #     block.append(self[str(name) + 'L'][buffer_index * config.BLOCKSIZE:
    #                                        (buffer_index + 1) * config.BLOCKSIZE])
    #     block.append(self[str(name) + 'R'][buffer_index * config.BLOCKSIZE:
    #                                        (buffer_index + 1) * config.BLOCKSIZE])
    #     return block


    def get_sample(self, name):
        return self.samples[name].get_sample()


def play_on_buffer(bufname, data, sample):
    assert isinstance(data, Data)

    index = 0
    while index < len(sample) - 1:
        if data.buffer_is_full(bufname):
            time.sleep(config.SLEEPTIME)
            continue

        data.put_block(
            bufname,
            sample[index:index+config.BLOCKSIZE,0],
            sample[index:index+config.BLOCKSIZE,1])
        index += config.BLOCKSIZE

