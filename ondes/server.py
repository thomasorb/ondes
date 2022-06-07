import sounddevice as sd
import soundfile as sf
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


class RandomWalker(object):
    
    def __init__(self, dimx, dimy, ix, iy, stepsize, gravpower=2):

        self.gravpower = gravpower
        self.dimx = dimx
        self.dimy = dimy
        self.field = np.zeros((dimx, dimy))
        self.grav_center = int(ix), int(iy)
        self.ix, self.iy = int(ix), int(iy)
        self.stepsize = 21
        self.last_irand = None
        self.last_p = None


    def get_radius(self, x, y):
        return np.sqrt(np.sum((np.array([x, y]).T - np.array(self.grav_center))**2, axis=-1))
    
    def get_proba(self, x, y):
        xs, ys = np.mgrid[x-self.stepsize//2:x+self.stepsize//2+1,
                          y-self.stepsize//2:y+self.stepsize//2+1,]
        r = self.get_radius(xs.flatten(), ys.flatten()).reshape((self.stepsize, self.stepsize))
        proba = 1 / (r**self.gravpower) # asymptotic freedom like strong interaction
        proba -= proba.min()
        proba[np.isinf(proba)] = 0
        proba /= np.sum(proba)
        return xs.flatten(), ys.flatten(), proba.flatten()

    def get_next(self):
        self.field[self.ix, self.iy] = 1
        xs, ys, p = self.get_proba(self.ix, self.iy)
        
        if self.last_irand is not None:
            p[self.last_irand] *= 300

        p[(xs == self.ix)*(ys == self.iy)] = 0
        p[xs >= self.dimx] = 0
        p[xs < 0] = 0
        p[ys >= self.dimy] = 0
        p[ys < 0] = 0
        p = p.flatten()
            
        p /= np.sum(p)
        irand = np.random.choice(np.arange(self.stepsize**2), size=1, p=p)
        self.ix, self.iy = xs[irand], ys[irand]
        self.last_irand = int(irand)
        self.last_p = p.copy()
        return int(self.ix), int(self.iy)

    def set_grav_center(self, ix, iy, fast_move=False):
        if (int(ix), int(iy)) == self.grav_center:
            return
        self.grav_center = int(ix), int(iy)
        if fast_move:
            self.ix, self.iy = int(ix), int(iy)
        
class Server(object):

    def __init__(self, data, file_path, file_srate):
        assert isinstance(data, core.Data)
        self.data = data
        self.recording = False
        
        # if 2D, must have shape (samples, channels)
        self.basedata = np.abs(np.load(file_path).astype(config.DTYPE))
        logging.info('data shape: {}'.format(self.basedata.shape))
        logging.info('data type: {}'.format(self.basedata.dtype))

        self.low_timing_resolution = False
        self.rw = None
        if self.basedata.ndim == 3: # sitelle data

            x_init, y_init = self.basedata.shape[0]//2, self.basedata.shape[1]//2
            self.data['x_orig0'].set(x_init)
            self.data['y_orig0'].set(y_init)
            self.rw = RandomWalker(self.basedata.shape[0],
                                   self.basedata.shape[1],
                                   x_init, y_init, 11) # step size
            self.low_timing_resolution = True
            self.filelen = 10
            self.nchans = 512
            self.depth = 1

            self.bins = np.linspace(0, self.basedata.shape[2]-1, self.nchans//self.depth+1).astype(int)
            self.bins_diff = np.diff(self.bins)
            self.filedata = np.empty((self.filelen, self.nchans), dtype=config.DTYPE)

            # load first N samples
            for i in range(self.filelen - 1):
                self.filedata[i,:] = self.get_next()
            # last sample == first sample for fast interpolation reasons
            self.filedata[-1,:] = self.filedata[0,:] 
            
        elif self.basedata.ndim == 2: # radio data
            self.filedata = self.basedata
            if self.filedata.shape[1] > 128:
                self.low_timing_resolution = True
            self.filedata /= np.nanpercentile(self.filedata[:100,:], 99)

            # invert channels
            self.filedata = self.filedata[:,::-1]
            # cut to blocksize
            #self.filedata = self.filedata[:self.filedata.shape[0] - (self.filedata.shape[0] % config.BLOCKSIZE),:]
            #logging.info('data shape (cut to blocksize): {}'.format(self.filedata.shape))
        
            self.nchans = int(self.filedata.shape[1])
            self.filelen = self.filedata.shape[0]
            
        else: raise Exception('bad input data')

        if self.low_timing_resolution:
            logging.warning('low timing resolution')
        #assert self.filedata.shape[1] <= 128 # else, wave carriers will give have the same frequency since they are divided in 127 notes
            
        self.sampling_rate_factor = config.SAMPLERATE / file_srate            
        
        zerosdata = np.zeros((config.BLOCKSIZE, 2), dtype=config.DTYPE)
            
        self.time_index = 0
        self.data_index = 0
        self.last_loaded = 0
        self.next_to_load = 0
        
        # init time matrix
        self.time_matrix = np.ones(
            (config.BLOCKSIZE, self.nchans),
            dtype=config.DTYPE) * np.arange(config.BLOCKSIZE).reshape((config.BLOCKSIZE, 1))

        def callback(outdata, frames, timing, status):

            def compute_interp_axis(data_index, sampling_rate_factor):
                interp_axis = np.arange(config.BLOCKSIZE, dtype=float) / sampling_rate_factor + data_index
                interp_axis = interp_axis % (self.filelen - 1)
                next_data_index = float(interp_axis[-1]) + (1 / sampling_rate_factor)
                return interp_axis, next_data_index

            stime = time.time()
            
            assert frames == config.BLOCKSIZE                


            cc_notemin = self.data['cc0'].get()
            cc_noterange = self.data['cc1'].get()
            cc_srate = self.data['cc2'].get()
            cc_bright = self.data['cc3'].get()
            #cc_lowpass = self.data['cc4'].get()
            #cc_pink = self.data['cc5'].get()
            cc_chanc = self.data['cc6'].get()
            cc_chanstd = self.data['cc7'].get()
            #cc_harm_n = self.data['cc78'].get()
            #cc_harm_step = self.data['cc34'].get()
            #cc_harm_level = self.data['cc35'].get()
            cc_volume = self.data['cc16'].get()
            cc_rec = self.data['cc45'].get()
            # get output
            if status.output_underflow:
                logging.warn('Output underflow: increase blocksize?')
                data = np.copy(zerosdata)
            if status:
                logging.warn('callback status')
                data = np.copy(zerosdata)
            else:
                
                if self.rw is not None:                
                    self.rw.set_grav_center(
                        self.data['x_orig0'].get(),
                        self.data['y_orig0'].get(),
                        fast_move=True)
                
                #cc_fold_n = self.data['cc33'].get()
                #cc_fold_delay = self.data['cc34'].get()
                
                #print(cc_notemin, cc_noterange, cc_srate, cc_bright, cc_lowpass, cc_pink, cc_chanc, cc_chanstd)
                try:
                    
                    # channels selection
                    chanc = utils.cc_rescale(cc_chanc, 0, self.nchans - 1)
                    chanstd = utils.cc_rescale(cc_chanstd, 0, self.nchans*2)
                    chanmin = int(max(chanc - 2*chanstd, 0))
                    chanmax = int(min(chanc + 2*chanstd + 1, self.nchans))
                    
                    nchans = chanmax - chanmin
                    chan_eq = np.exp(-(np.arange(self.nchans) - chanc)**2/(chanstd+0.5)**2)

                    # frequency range
                    notemin = utils.cc_rescale(cc_notemin, 0, 127)
                    noterange = utils.cc_rescale(cc_noterange, 0, 127)
                    notemax = min(notemin + noterange, 127)

                    # compute interp axis
                    sampling_rate_factor = self.sampling_rate_factor * 10**utils.cc_rescale(
                        cc_srate, -1, 1)
                    
                    interp_axis, self.data_index = compute_interp_axis(
                        self.data_index, sampling_rate_factor)

                    
                    notes = np.linspace(notemin, notemax, self.nchans)
                    freqs = utils.note2f(notes, config.A_MIDIKEY).reshape((1, self.nchans))

                    # load next sample
                    if self.rw is not None:
                        self.next_to_load = int(self.data_index) + 2
                        
                        if self.next_to_load == self.filelen:
                            self.next_to_load = 1
                        
                        if self.last_loaded != self.next_to_load:
                            self.filedata[self.next_to_load,:] = self.get_next()
                            if self.next_to_load == self.filelen - 1:
                                # copy last sample to first sample
                                self.filedata[0,:] = self.filedata[self.next_to_load,:]
                            self.last_loaded = int(self.next_to_load)
                            
                    # if self.phase.shape[0] > 1:
                    #     if self.low_timing_resolution:
                    #         phase_block = utils.fastinterp1d(
                    #             self.phase[:,chanmin:chanmax], interp_axis[:2])[0,:]
                    #     else:
                    #         phase_block = utils.fastinterp1d(
                    #             self.phase[:,chanmin:chanmax], interp_axis)
                    #     phase_block = self.phase[0,chanmin:chanmax]
                    # else:
                    #     phase_block = self.phase[:,chanmin:chanmax]

                    carrier_matrix = np.cos((self.time_matrix[:,chanmin:chanmax] + self.time_index) * (freqs[:, chanmin:chanmax] * (2 * np.pi / config.SAMPLERATE)))# + phase_block)
                    
                    if self.low_timing_resolution:
                        filedata_block = utils.fastinterp1d(
                            self.filedata[:,chanmin:chanmax], interp_axis[:2])[0,:]
                    else:
                      filedata_block = utils.fastinterp1d(
                          self.filedata[:,chanmin:chanmax], interp_axis)  
                        
                    # brightness
                    brightness = utils.cc_rescale(cc_bright, 0, 10)
                    filedata_block = filedata_block ** brightness

                    # equalizer
                    #noct = np.log(fmax/fmin) / np.log(2) # number of octaves
                    #pink_factor = utils.cc_rescale(cc_pink, -4, 0)
                    #pink = (np.linspace(1, noct, self.nchans)**(pink_factor))

                    carrier_matrix *= (filedata_block * chan_eq[chanmin:chanmax]) #* pink[chanmin:chanmax]
                    
                    # merge channels
                    sound = np.mean(carrier_matrix, axis=1)
                    
                    # lowpass
                    # lowpass = utils.cc_rescale(cc_lowpass, 0, 50)
                    # if lowpass > 0:
                    #     sound = np.array(ccore.fast_lowp(sound.astype(np.float32), lowpass))
                    
                    # volume
                    sound *= 10**utils.cc_rescale(cc_volume, -2, 2)

                    # clip
                    sound = np.clip(sound, -1, 1)
                    
                    data = np.zeros((config.BLOCKSIZE, 2), dtype=config.DTYPE)
                    data[:,0] = sound
                    data[:,1] = data[:,0]
                    
                except Exception as e:
                    logging.warn('callback exception: {}'.format(e))
                    data = np.copy(zerosdata)

                self.time_index += config.BLOCKSIZE

                if self.rw is not None:
                    try:
                        self.data['display_spectrum0'][:int(self.nchans)] = np.array(
                            filedata_block, dtype=config.DTYPE)
                        self.data['display_spectrum_len0'].set(int(self.nchans))
                    except Exception as e:
                        logging.warning('error at display', e)

            data = data.astype(config.DTYPE)
            
            #assert data.shape == outdata.shape, 'data len must match'

            # morphing with input

            #data = utils.morph(indata, data, 0.5)
            
            # send to out channel
            
            outdata[:] = data

            if cc_rec and not self.recording:
                self.recording = True
                self.to_record = list()
                
            if not cc_rec and self.recording:
                self.recording = False
                if len(self.to_record) > 0:
                    sf.write('{}.wav'.format(int(time.time())),
                             np.vstack(self.to_record),
                             config.SAMPLERATE)
                self.to_record = list()
                
            if self.recording:
                self.to_record.append(data)
                
                

            #print(time.time() - stime, config.BLOCKSIZE/config.SAMPLERATE)
            self.data.timing_buffers['server_callback_time'].put(time.time() - stime)
            
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

    def get_next(self):
        if self.rw is None: raise Exception('random walker not initalised')
        vall = list()
        for idepth in range(self.depth):
            iix, iiy = self.rw.get_next()
            v = self.basedata[iix,iiy,:]
            vbin = np.add.reduceat(np.abs(v), self.bins)[:-1] / self.bins_diff
            vbin /= np.max(vbin)
            vall.append(vbin)
        
        self.data['x0'].set(iix)
        self.data['y0'].set(iiy)
        return np.hstack(vall)
            
