import sounddevice as sd
import logging
import multiprocessing as mp
import queue
import numpy as np
import time
import scipy.interpolate

from . import core
from . import config
from . import utils
from . import ccore

class Server(object):

    def __init__(self, data, file_path, file_srate):
        assert isinstance(data, core.Data)
        self.data = data

        # if 2D, must have shape (samples, channels)
        self.filedata = np.load(file_path).astype(config.DTYPE)
        logging.info('data shape: {}'.format(self.filedata.shape))
        logging.info('data type: {}'.format(self.filedata.dtype))

        self.sampling_rate_factor = config.SAMPLERATE / file_srate

        # invert channels
        self.filedata = self.filedata[:,::-1]
            
        self.filedata /= np.max(self.filedata)

        self.filedata = self.filedata[:self.filedata.shape[0] - (self.filedata.shape[0] % config.BLOCKSIZE),:]
        logging.info('data shape (cut to blocksize): {}'.format(self.filedata.shape))
        
        
        
        self.nchans = int(self.filedata.shape[1])


        zerosdata = np.zeros((config.BLOCKSIZE, 2), dtype=config.DTYPE)
            
        self.time_index = 0
        self.data_index = 0
        
        # init time matrix
        self.time_matrix = np.zeros((config.BLOCKSIZE, self.nchans), dtype=config.DTYPE)
        for ichan in range(self.nchans):
            self.time_matrix[:,ichan] = np.arange(config.BLOCKSIZE) + np.random.uniform() * config.SAMPLERATE / config.FMIN
            
        def callback(outdata, frames, timing, status):

            def compute_interp_axis(data_index, sampling_rate_factor):
                interp_axis = np.arange(config.BLOCKSIZE, dtype=float) / sampling_rate_factor + data_index
                interp_axis = interp_axis % (self.filedata.shape[0] - 1)
                next_data_index = float(interp_axis[-1]) + (1 / sampling_rate_factor)
                return interp_axis, next_data_index

            stime = time.time()
            
            assert frames == config.BLOCKSIZE                
                
            # get output
            if status.output_underflow:
                logging.warn('Output underflow: increase blocksize?')
                data = np.copy(zerosdata)
            if status:
                logging.warn('callback status')
                data = np.copy(zerosdata)
            else:
                
                cc_fmin = self.data['cc1'].get()
                cc_frange = self.data['cc2'].get()
                cc_srate = self.data['cc3'].get()
                cc_bright = self.data['cc4'].get()
                cc_lowpass = self.data['cc5'].get()
                cc_pink = self.data['cc6'].get()
                cc_chanc = self.data['cc7'].get()
                cc_chanstd = self.data['cc8'].get()
                #cc_fold_n = self.data['cc33'].get()
                #cc_fold_delay = self.data['cc34'].get()
                
                #print(cc_fmin, cc_frange, cc_srate, cc_bright, cc_lowpass, cc_pink, cc_chanc, cc_chanstd)
                
                try:
                    # channels selection
                    chanc = utils.cc_rescale(cc_chanc, 0, self.nchans - 1)
                    chanstd = utils.cc_rescale(cc_chanstd, 0, self.nchans*2)
                    chanmin = int(max(chanc - 2*chanstd, 0))
                    chanmax = int(min(chanc + 2*chanstd + 1, self.nchans))
                    
                    nchans = chanmax - chanmin
                    chan_eq = np.exp(-(np.arange(self.nchans) - chanc)**2/(chanstd+0.5)**2)

                    # frequency range
                    fmin = 20 * 10**utils.cc_rescale(cc_fmin, 0, 3)
                    frange = 20 * 10**utils.cc_rescale(cc_frange, 0, 3)
                    fmax = min(fmin + frange, config.FMAX)

                    # compute carrier waves
                    carrier_matrix = np.copy(self.time_matrix[:,chanmin:chanmax]) + self.time_index
                    freqs = 10**np.linspace(np.log10(fmin), np.log10(fmax), self.nchans).reshape((1, self.nchans))
                    carrier_matrix *= 2 * np.pi * freqs[:, chanmin:chanmax] / config.SAMPLERATE 

                    carrier_matrix = np.cos(carrier_matrix)

                    sampling_rate_factor = self.sampling_rate_factor * 10**utils.cc_rescale(cc_srate, -1, 1)
                    
                    interp_axis, self.data_index = compute_interp_axis(self.data_index, sampling_rate_factor)
                    
                    filedata_block = utils.fastinterp1d(
                        self.filedata[:,chanmin:chanmax], interp_axis)

                    # folding
                    # fold_n = int(utils.cc_rescale(cc_fold_n, 0, 60))
                    # fold_delay = utils.cc_rescale(cc_fold_delay, 40, 50) # ms
                    # fold_delay *= config.SAMPLERATE / sampling_rate_factor / 1000. # data samples
                    # #next_data_index = float(self.data_index)
                    # for i in range(fold_n):
                    #     iinterp_axis, _ = compute_interp_axis(self.data_index + fold_delay * (i + 1), sampling_rate_factor)
                    #     filedata_block += utils.fastinterp1d(
                    #         self.filedata[:,chanmin:chanmax], iinterp_axis)
                    # filedata_block /= fold_n + 1


                    # brightness
                    brightness = utils.cc_rescale(cc_bright, 0, 10)
                    filedata_block = filedata_block ** brightness

                    # matrix prod
                    carrier_matrix *= filedata_block

                    # equalizer
                    #noct = np.log(fmax/fmin) / np.log(2) # number of octaves
                    #pink_factor = utils.cc_rescale(cc_pink, -4, 0)
                    #pink = (np.linspace(1, noct, self.nchans)**(pink_factor))

                    carrier_matrix *= chan_eq[chanmin:chanmax] #* pink[chanmin:chanmax]
                    
                    # merge channels
                    sound = np.mean(carrier_matrix, axis=1).astype(np.float32)

                    # lowpass
                    lowpass = utils.cc_rescale(cc_lowpass, 0, 50)
                    if lowpass > 0:
                        sound = ccore.fast_lowp(sound, lowpass)
                    
                    
                    data = np.zeros((config.BLOCKSIZE, 2), dtype=config.DTYPE)
                    data[:,0] = sound
                    data[:,1] = data[:,0]
                    
                except Exception as e:
                    logging.warn('callback exception: {}'.format(e))
                    data = np.copy(zerosdata)

                self.time_index += config.BLOCKSIZE
                    
                
            data = data.astype(config.DTYPE)
            
            #assert data.shape == outdata.shape, 'data len must match'

            # morphing with input

            #data = utils.morph(indata, data, 0.5)
            
            # send to out channel
            outdata[:] = data

            #print(time.time() - stime, config.BLOCKSIZE/config.SAMPLERATE)
            #self.data.timing_buffers['server_callback_time'].put(time.time() - stime)
            return

        self.stream = sd.OutputStream(
            samplerate=config.SAMPLERATE, blocksize=config.BLOCKSIZE,
            device=self.get_device(), channels=2, dtype=config.DTYPE,
            callback=callback)
            
        self.stream.start()
        
        timeout = config.BLOCKSIZE * config.BUFFERSIZE / config.SAMPLERATE

        while True:
            time.sleep(config.SLEEPTIME)
            
        
    def __del__(self):
        try:
            self.stream.close()
        except Exception as e:
            pass


    def get_device(self):
        devices = sd.query_devices()
        logging.info('audio devices: \n{}'.format(repr(devices)))
        
        for device in devices:
            if config.DEVICE in device['name']:
                print(device)
                return device['name']

        logging.warning('device not found switching to default')
        return 'default'
