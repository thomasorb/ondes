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
from . import maths

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
            self.maxchans = self.basedata.shape[2] - 1
            self.nchans = config.NCHANNELS
            logging.info('channels number: {}'.format(self.maxchans))
            self.depth = 1

            self.bins = np.linspace(0, self.basedata.shape[2]-1, self.maxchans//self.depth+1).astype(int)
            self.bins_diff = np.diff(self.bins)
            self.filedata = np.empty((self.filelen, self.maxchans), dtype=config.DTYPE)

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
        
            self.maxchans = int(self.filedata.shape[1])
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
        self.notes = np.zeros((127, 4), dtype=float)
        
        # init time matrix
        self.time_matrix = np.ones(
            (config.BLOCKSIZE, self.maxchans),
            dtype=config.DTYPE) * np.arange(config.BLOCKSIZE).reshape((config.BLOCKSIZE, 1)).astype(config.DTYPE)
                    
        def callback(outdata, frames, timing, status):

            def compute_interp_axis(data_index, sampling_rate_factor):
                interp_axis = np.arange(config.BLOCKSIZE, dtype=float) / sampling_rate_factor + data_index
                interp_axis = interp_axis % (self.filelen - 1)
                next_data_index = float(interp_axis[-1]) + (1 / sampling_rate_factor)
                return interp_axis, next_data_index

            stime = time.time()
            
            assert frames == config.BLOCKSIZE                


            cc_freqmin = self.data['cc_freqmin'].get()
            cc_freqrange = self.data['cc_freqrange'].get()
            cc_srate = self.data['cc_srate'].get()
            cc_bright = self.data['cc_bright'].get()
            #cc_lowpass = self.data['cc20'].get()
            #cc_pink = self.data['cc21'].get()
            #cc_chanc = self.data['cc_chanc'].get()
            #cc_chanstd = self.data['cc_chanstd'].get()
            #cc_harm_n = self.data['cc33'].get()
            #cc_harm_step = self.data['cc34'].get()
            cc_comp_threshold = self.data['cc_comp_threshold'].get()
            cc_comp_level = self.data['cc_comp_level'].get()
            cc_release_time = self.data['cc_release_time'].get()
            cc_attack_time = self.data['cc_attack_time'].get()
            cc_volume = self.data['cc_volume'].get()
            cc_rec = self.data['cc_rec'].get()
            

            release_time = utils.cc_rescale(cc_release_time, 1, config.MAX_RELEASE_TIME) # in ms
            attack_time = utils.cc_rescale(cc_attack_time, 1, config.MAX_ATTACK_TIME) # in ms
            
            for inote in range(len(self.notes)):
                if self.data['note{}'.format(inote)].get() > 0:
                    if self.notes[inote, 3]:
                        if self.notes[inote, 1] > 0: # note released
                            # inote: vel, release_time, attack_time
                            self.notes[inote, :] = np.array((self.data['vel{}'.format(inote)].get(), 0, 0, 1))
                        else:
                            self.notes[inote, 2] += config.BLOCKTIME # add time to attack
                    
                            
                    else:
                        self.notes[inote, :] = np.array((self.data['vel{}'.format(inote)].get(), 0, 0, 1))
                elif self.notes[inote, 3]:
                    if self.notes[inote, 1] > release_time:
                        self.notes[inote, 3] = 0
                    else:
                        self.notes[inote, 1] += config.BLOCKTIME # add time to release
                    
                    
            #for inote in self.notes_old():
            #    if
            #self.notes_old = list(self.notes) # copy of the list to compute release
            
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
                
                #print(cc_freqmin, cc_freqrange, cc_srate, cc_bright, cc_lowpass, cc_pink, cc_chanc, cc_chanstd)
                try:
                    
                    # channels selection
                    # chanc = utils.cc_rescale(cc_chanc, 0, self.maxchans - 1)
                    # chanstd = utils.cc_rescale(cc_chanstd, 0, self.maxchans*2)
                    # chanmin = int(max(chanc - 2*chanstd, 0))
                    # chanmax = int(min(chanc + 2*chanstd + 1, self.maxchans))
                    ## chan_eq = np.exp(-(np.arange(self.maxchans) - chanc)**2/(chanstd+0.5)**2)
                    chan_eq = np.ones(self.maxchans)
                    
                    # frequency range
                    freqmin = utils.cc_rescale(cc_freqmin, 0, 127)
                    freqrange = utils.cc_rescale(cc_freqrange, 0, 127)
                    freqmax = min(freqmin + freqrange, 127)

                    # compute interp axis
                    sampling_rate_factor = self.sampling_rate_factor * 10**utils.cc_rescale(
                        cc_srate, -1, 1)
                    
                    interp_axis, self.data_index = compute_interp_axis(
                        self.data_index, sampling_rate_factor)


                    if self.low_timing_resolution:
                        filedata_block = utils.fastinterp1d(
                            self.filedata[:,0:self.maxchans], interp_axis[:2])[0,:]
                    else:
                        filedata_block = utils.fastinterp1d(
                            self.filedata[:,0:self.maxchans], interp_axis)
                        
                    #freqsorder = np.arange(nchans)
                    #rng = np.random.default_rng()
                    #rng.shuffle(freqsorder)
                    freqsorder = np.argsort(filedata_block)[::-1][:self.nchans]
                    

                    notefreqs = np.linspace(freqmin, freqmax, self.maxchans)
                    freqs = utils.note2f(notefreqs, config.A_MIDIKEY).astype(config.DTYPE)

                    freqs = freqs[freqsorder].reshape((1, self.nchans))

             
                    
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
                    #             self.phase[:,0:self.nchans], interp_axis[:2])[0,:]
                    #     else:
                    #         phase_block = utils.fastinterp1d(
                    #             self.phase[:,0:self.nchans], interp_axis)
                    #     phase_block = self.phase[0,0:self.nchans]
                    # else:
                    #     phase_block = self.phase[:,0:self.nchans]



                    # compute carrier matrix

                            
                    ## important to keep the value computed by cos to float32 (much much faster)


                    comp_threshold = 10**(utils.cc_rescale(cc_comp_threshold, -5, -0.0001))
                    comp_level = 10**(utils.cc_rescale(
                        utils.cc_rescale(cc_comp_level, cc_comp_threshold, 127), -5, -0.0001))
                    
                    
                    carrier_matrix = np.zeros((self.time_matrix.shape[0], self.nchans))
                    
                    for inote in range(len(self.notes)):
                        if not self.notes[inote, 3]: continue
                        freqshift = utils.note2f(inote, config.A_MIDIKEY) / utils.note2f(config.BASENOTE, config.A_MIDIKEY)
                        ivelocity = 10**((self.notes[inote, 0] - 127)/(10*config.VELOCITY_SCALE))

                        iattack = np.linspace(self.notes[inote, 2],
                                              self.notes[inote, 2] + config.BLOCKTIME,
                                              config.BLOCKSIZE).reshape((config.BLOCKSIZE,1))
                        iattack /= attack_time
                        iattack = np.clip(iattack,0,1)

                        irelease = np.linspace(self.notes[inote, 1] - config.BLOCKTIME,
                                               self.notes[inote, 1],
                                               config.BLOCKSIZE).reshape((config.BLOCKSIZE,1))
                        irelease = (release_time - irelease) / release_time
                        irelease = np.clip(irelease,0,1)
                        
                        
                        carrier_matrix += (maths.compress(np.cos((self.time_matrix[:,0:self.nchans] + config.DTYPE(self.time_index)) * (freqs[:, 0:self.nchans] * freqshift * (2 * np.pi / config.SAMPLERATE))),
                                                          comp_threshold,
                                                          comp_level)
                                           * ivelocity * irelease * iattack)
                        
                                        
                    # brightness
                    brightness = config.DTYPE(utils.cc_rescale(cc_bright, 0, 10))
                    filedata_block = filedata_block ** brightness

                    # equalizer
                    #noct = np.log(fmax/fmin) / np.log(2) # number of octaves
                    #pink_factor = utils.cc_rescale(cc_pink, -4, 0)
                    #pink = (np.linspace(1, noct, self.maxchans)**(pink_factor))

                    carrier_matrix *= (filedata_block * chan_eq)[freqsorder] #* pink[0:self.nchans]
                    
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
                        self.data['display_spectrum0'][:self.nchans] = np.array(filedata_block[np.sort(freqsorder)], dtype=config.DTYPE)
                        self.data['display_spectrum_len0'].set(self.nchans)
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
                
            self.data.timing_buffers['server_callback_time'].put(time.time() - stime)
            
            return

        self.stream = sd.OutputStream(
            samplerate=config.SAMPLERATE, blocksize=config.BLOCKSIZE,
            latency=config.LATENCY / 1000,
            device=self.get_device(), channels=2, dtype=config.DTYPE,
            callback=callback)
            
        self.stream.start()
        

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
            
