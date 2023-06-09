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
from . import effects

class RandomWalker(object):
    
    def __init__(self, dimx, dimy, ix, iy, stepsize, gravpower=4):

        self.gravpower = gravpower
        self.dimx = dimx
        self.dimy = dimy
        self.field = np.zeros((dimx, dimy))
        self.grav_center = int(ix), int(iy)
        self.ix, self.iy = int(ix), int(iy)
        self.stepsize = stepsize
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
                                   x_init, y_init, config.RANDOMWALKER_RADIUS) # step size
            self.low_timing_resolution = True
            self.filelen = 10
            self.maxchans = self.basedata.shape[2] 
            self.nchans = config.NCHANNELS
            logging.info('channels number: {}'.format(self.maxchans))

            #self.bins = np.linspace(0, self.basedata.shape[2]-1, self.maxchans+1).astype(int)
            #self.bins_diff = np.diff(self.bins)
            self.filedata = np.zeros((self.filelen, self.maxchans), dtype=config.DTYPE)

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
        

        self.time_index = 0
        self.data_index = 0
        self.last_loaded = 0
        self.next_to_load = 0
        self.notes = np.zeros((127, 4), dtype=float)
        # self.trans = np.zeros((config.TRANS_SIZE, 4), dtype=float)
        # self.trans[:,0] = np.random.randint(0, self.filedata.shape[0])
        # self.trans[:,2] = 1000 * config.TRANS_RELEASE
        self.keep = None
        self.vL = 0
        self.pL = 0
        self.vR = 0
        self.pR = 0
        self.callback_rands = np.full(3, 1e-5)
        self.callback_rands_factor = np.random.uniform(2, 10, size=len(self.callback_rands))
        self.callback_rands_phase = np.random.uniform(1, 4, size=len(self.callback_rands))

        self.sample_rands = np.full(len(self.notes), 1e-5)
        self.sample_rands_factor = np.random.uniform(2, 10, size=len(self.sample_rands))
        self.sample_rands_phase = np.random.uniform(1, 4, size=len(self.sample_rands))
        
        # init time matrix
        self.time_matrix = np.ones(
            (config.BUFFERSIZE, self.maxchans),
            dtype=config.DTYPE) * np.arange(config.BUFFERSIZE).reshape((config.BUFFERSIZE, 1)).astype(config.DTYPE)

        
        def get_sample(blocksize, cc):

            sample_inhomogeneity = 10**utils.cc_rescale(cc['sample_inhomogeneity'], -5, 0)
            self.sample_rands = np.cos(self.time_index / config.SAMPLERATE / self.sample_rands_factor + self.sample_rands_phase) * sample_inhomogeneity
            
            def compute_interp_axis(data_index, sampling_rate_factor):
                interp_axis = np.arange(blocksize, dtype=float) / sampling_rate_factor + data_index
                interp_axis = interp_axis % (self.filelen - 1)
                next_data_index = float(interp_axis[-1]) + (1 / sampling_rate_factor)
                return interp_axis, next_data_index

            blocktime = blocksize / config.SAMPLERATE * 1000
            #print(blocksize, blocktime, np.sum(self.notes[:,3]))
            
            release_time = utils.cc_rescale(cc['release_time'], 1, config.MAX_RELEASE_TIME) # in ms
            attack_time = utils.cc_rescale(cc['attack_time'], 1, config.MAX_ATTACK_TIME) # in ms
            
            for inote in range(len(self.notes)):
                if self.data['note{}'.format(inote)].get() > 0:
                    if self.notes[inote, 3]:
                        if self.notes[inote, 1] > 0: # note released
                            # inote: vel, release_time, attack_time
                            self.notes[inote, :] = np.array((self.data['vel{}'.format(inote)].get(), 0, 0, 1))
                        else:
                            self.notes[inote, 2] += blocktime # add time to attack
                    
                            
                    else:
                        self.notes[inote, :] = np.array((self.data['vel{}'.format(inote)].get(), 0, 0, 1))
                elif self.notes[inote, 3]:
                    if self.notes[inote, 1] > release_time:
                        self.notes[inote, 3] = 0
                    else:
                        self.notes[inote, 1] += blocktime # add time to release

            
            if self.rw is not None:                
                self.rw.set_grav_center(
                    self.data['x_orig0'].get(),
                    self.data['y_orig0'].get(),
                    fast_move=True)
                
                    # channels selection
            # chanc = utils.cc_rescale(cc['chanc'], 0, self.maxchans - 1)
            # chanstd = utils.cc_rescale(cc['chanstd'], 0, self.maxchans*2)
            # chanmin = int(max(chanc - 2*chanstd, 0))
            # chanmax = int(min(chanc + 2*chanstd + 1, self.maxchans))
            ## chan_eq = np.exp(-(np.arange(self.maxchans) - chanc)**2/(chanstd+0.5)**2)
            chan_eq = np.ones(self.maxchans)

            # frequency range
            freqmin = utils.cc_rescale(cc['freqmin'], -64, 127+64)
            freqrange = utils.cc_rescale(cc['freqrange'], 0, 256)
            freqmax = freqmin + freqrange#, 256)

            notefreqs = np.linspace(freqmin, freqmax, self.maxchans)
            freqs = utils.note2f(notefreqs, config.A_MIDIKEY).astype(config.DTYPE)

            # compute interp axis
            sampling_rate_factor = self.sampling_rate_factor * 10**utils.cc_rescale(
                cc['srate'], -1, 3)

            interp_axis, self.data_index = compute_interp_axis(self.data_index, sampling_rate_factor)


            if self.low_timing_resolution:
                filedata_block = utils.fastinterp1d(
                    np.copy(self.filedata[:,0:self.maxchans]), interp_axis[:2])[0,:]
            else:
                filedata_block = utils.fastinterp1d(
                    np.copy(self.filedata[:,0:self.maxchans]), interp_axis)

            if cc['keep'] > 0:
                if self.keep is None:
                    self.keep = np.copy(filedata_block)
                else:
                    self.keep = (filedata_block + self.keep)/2
            elif self.keep is not None:
                filedata_block = (filedata_block + self.keep)/2

            if cc['unkeep'] > 0:
                self.keep = None

            # compute most significant freqs (witout transients)
            freqsorder = np.argsort(filedata_block)[::-1][:self.nchans]

            # # autotune
            # autotune = 0
            # basef = (freqmin + freqmax)/2.
            # if autotune:
            #     nearestf = np.min(np.abs(freqs[freqsorder][:10] - basef))
            #     freqs *= basef / nearestf


            # add random transients sounds
            # trans_release = config.TRANS_RELEASE * utils.cc_rescale(cc['trans_release'], 0.01, 1)

            # if np.any(self.trans[:,2] > trans_release) and np.random.uniform() < .1:
            #     self.trans = np.roll(self.trans, 1, axis=0)
            #     #self.trans[0, 0] = np.random.randint(0,filedata_block.shape[0])
            #     self.trans[0, 0] = freqsorder[np.random.randint(0, len(freqsorder))]

            #     self.trans[0, 1] = np.random.uniform(np.min(filedata_block),  + utils.cc_rescale(
            #         cc['trans_presence'], 0, 1) * np.max(filedata_block))

            #     self.trans[0, 2] = 0

            # self.trans[:,2] += blocksize / config.SAMPLERATE
            # filedata_block[self.trans[:,0].astype(int)] = self.trans[:,1] * np.clip(((trans_release - self.trans[:,2])/trans_release), 0, 1)

            ## recompute most significant freqs to capture transients
            # freqsorder = np.argsort(filedata_block)[::-1][:self.nchans]


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
            carrier_matrix = np.zeros((blocksize, self.nchans), dtype=config.DTYPE)

            # def triangle(x, cosx):
            #     sinx = np.sin(x) 
            #     atan2 = np.arctan2(sinx, cosx) / np.pi
            #     return (np.where(sinx > 0, atan2, -atan2) *2) - 1

            for inote in range(len(self.notes)):
                if not self.notes[inote, 3]: continue
                freqshift = utils.note2f(inote, config.A_MIDIKEY) / utils.note2f(config.BASENOTE, config.A_MIDIKEY) 
                ivelocity = 10**((self.notes[inote, 0] - 127)/(10*config.VELOCITY_SCALE))

                iattack = np.linspace(self.notes[inote, 2],
                                      self.notes[inote, 2] + blocktime,
                                      blocksize).reshape((blocksize,1))
                iattack /= attack_time
                iattack = np.clip(iattack,0,1)

                irelease = np.linspace(self.notes[inote, 1] - blocktime,
                                       self.notes[inote, 1],
                                       blocksize).reshape((blocksize,1))
                irelease = (release_time - irelease) / release_time
                irelease = np.clip(irelease,0,1)

                x = (self.time_matrix[:blocksize,0:self.nchans] + config.DTYPE(self.time_index)) * (freqs[:, 0:self.nchans] * freqshift * (2 * np.pi / config.SAMPLERATE))
                #x += inote * 40000 # dephase to avoid ZPD effects when time near 0
                cosx = np.cos(x) 
                #carrier_matrix += (triangle(x, cosx) + cosx) / 2 * ivelocity * irelease * iattack
                a = utils.cc_rescale(cc['square'], 0., 1.)
                #powx = np.where(cosx > 0, cosx**a, -(-cosx)**a)

                carrier_matrix += (np.sign(cosx) * a + cosx * (1-a)) * ivelocity * irelease * iattack

            if np.sum(self.notes[:,3]) == 0:
                # security clean of time index since it
                # becomes too large and the precision of the
                # time matrix is only float32, results in a
                # gltich every 100 s. float32 have 7 digits of
                # precision max, which means no more than 200s
                # at 44100
                self.time_index = np.random.uniform(100000, 200000)




            # brightness
            brightness = config.DTYPE(utils.cc_rescale(cc['bright'], 0, 10))
            filedata_block = filedata_block ** brightness

            # equalizer
            #noct = np.log(fmax/fmin) / np.log(2) # number of octaves
            #pink_factor = utils.cc_rescale(cc['pink'], -4, 0)
            #pink = (np.linspace(1, noct, self.maxchans)**(pink_factor))

            carrier_matrix *= (filedata_block * chan_eq)[freqsorder] #* pink[0:self.nchans]

            # merge channels
            sound = np.mean(carrier_matrix, axis=1)

            
            # volume

            ## volume of each note is divided to avoid audio > 1 when multiple notes are stacked together
            sound /= config.POLYPHONY_VOLUME_ADJUST

            return sound, filedata_block, freqsorder
        

                    
        def callback(outdata, frames, timing, status):
            stime = time.time()
            blocksize = frames


            #get Control Change
            cc = dict()        
            for icc in config.CC_MATRIX:
                cc[icc] = self.data['cc_' + icc].get()
                
            # hacks##############
            #cc['volume'] = 126
            #cc['trans_presence'] = 0
            #cc['trans_release'] = 64            
            cc['rec'] = 0
            cc['keep'] = 0
            cc['unkeep'] = 0
            #######################
            
            filedata_block = None
            freqsorder = None
            
            # get output
            if status.output_underflow:
                logging.warn('Output underflow: increase blocksize?')
                sample = np.zeros(blocksize, dtype=config.DTYPE)
            if status:
                logging.warn('callback status')
                sample = np.zeros(blocksize, dtype=config.DTYPE)
            else:
                try:
                    sample, filedata_block, freqsorder = get_sample(blocksize, cc)
                except Exception as e:
                    logging.warn('get_sample() exception: {}'.format(e))
                    sample = np.zeros(blocksize, dtype=config.DTYPE)

                self.time_index += blocksize
                
                # security clean of time index since it becomes too large and the precision of the time matrix is only float32, results in a gltich every 100 s. float32 have 7 digits of precision max, which means no more than 200s at 44100
                if self.time_index > 100 * config.SAMPLERATE: self.time_index = 0
                
                if self.rw is not None and freqsorder is not None:
                    try:
                        self.data['display_spectrum0'][:self.maxchans] = np.array(filedata_block, dtype=config.DTYPE)
                        self.data['display_spectrum_len0'].set(self.maxchans)
                        self.data['display_scatterx0'][:self.nchans] = np.arange(self.maxchans, dtype=config.DTYPE)[freqsorder]
                        self.data['display_scattery0'][:self.nchans] = np.array(filedata_block[freqsorder], dtype=config.DTYPE)
                        self.data['display_scatter_len0'].set(self.nchans)
                    except Exception as e:
                        logging.warning('error at display', e)

            

            if cc['rec'] and not self.recording:
                self.recording = True
                self.to_record = list()
                
            if not cc['rec'] and self.recording:
                self.recording = False
                if len(self.to_record) > 0:
                    sf.write('{}.wav'.format(int(time.time())),
                             np.vstack(self.to_record),
                             config.SAMPLERATE)
                self.to_record = list()
                
            if self.recording:
                self.to_record.append(data)
                
            self.data.timing_buffers['server_callback_time'].put(time.time() - stime)


            # use next sample to generate effects larger than one sample (e.g. lowpass)

            # both = np.concatenate((self.next_sample, data))

            # # lowpass
            # lowpass = utils.cc_rescale(cc['lowpass'], 20, 1)
            # if lowpass > 1:
            #     #both[:,0] = np.array(ccore.fast_lowp(both[:,0].astype(np.float32), lowpass))
            #     #both[:,1] = np.array(ccore.fast_lowp(both[:,1].astype(np.float32), lowpass))
            #     both = effects.lowpass(both, int(lowpass))

            # # only the first part of both is kept
            # outdata[:] = both[:blocksize,:]
            # self.next_sample = both[blocksize:,:]

            
            # sample goes through elastic output
            m = 10 ** utils.cc_rescale(cc['m'], 0, 4)
            #omega0 = 10**utils.cc_rescale(cc['omega0'], -4, 0)
            damp = 0.5 * 10**utils.cc_rescale(cc['damp'], -4, 0)

            _max = np.max(sample)
            inhomogeneity = 10**utils.cc_rescale(cc['inhomogeneity'], -5, 0)

            self.callback_rands = np.cos(self.time_index / config.SAMPLERATE / self.callback_rands_factor + self.callback_rands_phase) * inhomogeneity
            
            sampleL, self.vL, self.pL = ccore.elastic_output(
                sample,
                m * (1 + self.callback_rands[0]),
                damp * (1 + self.callback_rands[1]),
                float(self.vL), float(self.pL))
            
            sampleR, self.vR, self.pR = ccore.elastic_output(
                sample,
                m * (1 - self.callback_rands[0]),
                damp * (1 - self.callback_rands[1]),
                float(self.vR), float(self.pR))

            stereo = np.empty((blocksize, 2), dtype=config.DTYPE)
            stereo[:,0] = np.array(sampleL)
            stereo[:,1] = np.array(sampleR)
            
            # volume
            stereo *= 10**utils.cc_rescale(cc['volume'], -2, 4)

            # compress
            comp_threshold = 10**(utils.cc_rescale(cc['comp_threshold'], -5, -0.0001))
            comp_level = 10**(utils.cc_rescale(
                utils.cc_rescale(cc['comp_level'], cc['comp_threshold'], 127), -5, -0.0001))

            stereo = maths.compress(stereo, comp_threshold, comp_level)

            # sigmoid (tube :D)
            tube = utils.cc_rescale(cc['tube'], 1, 10)
            stereo[:,0] = 2*(scipy.special.expit(stereo[:,0] * tube * (1 + self.callback_rands[2])) - 0.5)
            stereo[:,1] = 2*(scipy.special.expit(stereo[:,0] * tube * (1 - self.callback_rands[2])) - 0.5)

            # clip
            stereo = np.clip(stereo, -1, 1)

            outdata[:] = stereo

            return

        self.stream = sd.OutputStream(
            samplerate=config.SAMPLERATE, blocksize=0,#config.BLOCKSIZE,
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
        iix, iiy = self.rw.get_next()
        v = self.basedata[iix-config.BINNING:iix+config.BINNING+1,
                          iiy-config.BINNING:iiy+config.BINNING+1,:]
        v = np.mean(v, axis=(0,1))
        v /= np.max(np.abs((v)))
        #vbin = np.add.reduceat(np.abs(v), self.bins)[:-1] / self.bins_diff
        #vbin /= np.max(vbin)
        #vall.append(vbin)
        
        self.data['x0'].set(iix)
        self.data['y0'].set(iiy)
        return v #np.hstack(vall)
            
