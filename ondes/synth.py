import numpy as np
import logging
import sounddevice as sd
import scipy.interpolate

import time

from . import core
from . import ccore
from . import utils
from . import config
from . import effects
from . import display
from . import sampler


class Params(dict):
    """Special dictionary which elements can be accessed like
    attributes.
    """
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__
    __setattr__ = dict.__setitem__
    

class Param(object):

    def __init__(self, val, vmin, vmax, step, cast=int):
        self.cast = cast
        self.vmin = self.cast(vmin)
        self.vmax = self.cast(vmax)
        self.val = self.cast(val)
        self.step = self.cast(step)
        self.widget = None
        self.fig = None
        
    def set(self, val):
        self.val = self.cast(np.clip(val, self.vmin, self.vmax))
        
    # def toggle(self, event):
    #     self.val = not bool(self.val)
    #     if self.val:
    #         self.widget.ax.set_facecolor('tab:green')
    #         self.widget.color = 'tab:green'
    #     else:
    #         self.widget.ax.set_facecolor('tab:orange')
    #         self.widget.color = 'tab:orange'
    #     self.fig.canvas.draw_idle()
            
    def __call__(self):
        return self.cast(self.val)


class Wave(object):

    def __init__(self, mode='square'):

        if mode == 'sine':
            samplef = utils.sine
        else:
            samplef = utils.square
        
        sampleL = samplef(utils.note2f(0, config.A_MIDIKEY),
                          5000, config.SAMPLERATE)
        sampleR = np.copy(sampleL)
                
        start = ccore.get_first_zero(sampleL)
        end = ccore.get_last_zero(sampleL)
        sampleR = sampleR[start:end]
        sampleL = sampleL[start:end]
        
        sample = np.array((sampleL, sampleR)).T
        
        sample /= config.VOLUME_ADJUST
        self.downsample = sample

    @staticmethod
    def special_treatment(sampleL, sampleR):
        return sampleL, sampleR
    
    def apply(self, effect, *args):
        sample = getattr(effects, effect)(self.downsample, *args)
        if len(sample)%2:
            sample = np.concatenate((sample, np.zeros((1, sample.shape[1]), dtype=sample.dtype)))
        self.downsample = sample

    
class FakeSynth(object):

    def __init__(self, data):
        assert isinstance(data, core.Data)
        self.data = data
        self.name = 's0'
        
        mode = 'sine'
        while True:
            wave = np.copy(Wave(mode=mode).downsample)
            self.data.set_sample(self.name, wave)
            if mode == 'sine': mode = 'square'
            else: mode = 'sine'
            time.sleep(1)



    
class CubeSynth(object):


    def __init__(self, index, data, cubepath, dfpath):

        assert isinstance(data, core.Data)
        self.data = data
        self.name = 's{}'.format(int(index))
        
        cube = np.load(cubepath, mmap_mode='r')
        logging.info('cube shape: {}'.format(cube.shape))
        
        self.data['x_orig{}'.format(int(index))].set(config.XY[index][0])
        self.data['y_orig{}'.format(int(index))].set(config.XY[index][1])
        
        p = {
            'perc':Param(99.5, 95., 100., 0.05, cast=float),
            'depth':Param(4, 1, 10, 1),
            'deriv':Param(2, 1, 10, 1),
            'r':Param(1, 1, 30, 1),
            'reso':Param(4.8, 0.5, 30, 0.1, cast=float),
            'eq_power':Param(-2, -5, 5, 0.1, cast=float),
            'innerpad':Param(True, False, True, None, cast=bool),
            'fmin':Param(0, 0, 500, 1, cast=float),
            'frange':Param(1100, 200, 5000, 10, cast=float),
            'volume':Param(-1.6, -4, 1, 0.01, cast=float),
            'duration':Param(config.SYNTH_LOOP_TIME, 1, 30, 0.1, cast=float),
            'loops':Param(1, 1, 10, 1),
            'spectrum_roll':Param(0, -1, 1, 0.01, cast=float),
            'sample_roll':Param(0, -1, 1, 0.01, cast=int),
            'note':Param(62, 0, 127, 1),
        }
        self.p = Params(p)


        last_spectrum = None
        while True:
            stime = time.time()
            
            data = list()

            r = int(self.p.r())
                
            for iloop in range(self.p.depth()):
                #rx, ry = np.random.randint(-self.p.deriv(), self.p.deriv(), 2)
                self.x = int(self.data['x_orig{}'.format(int(index))].get() + np.random.standard_normal() * self.p.deriv())
                self.y = int(self.data['y_orig{}'.format(int(index))].get() + np.random.standard_normal() * self.p.deriv())
                _ispec = np.sum(cube[self.x-r:self.x+r+1,
                                     self.y-r:self.y+r+1, :], axis=(0,1))
                #_ispec -= np.min(_ispec)
                data.append(_ispec)
                
                
            new_spectrum = np.concatenate(data)
            if last_spectrum is not None:
                spectrum = new_spectrum + last_spectrum
            else:
                spectrum = new_spectrum
            last_spectrum = np.copy(new_spectrum)
            
            spectrum_to_draw = np.copy(spectrum)
            spectrum_to_draw = spectrum_to_draw[:min(config.MAX_DISPLAY_SIZE,
                                                     len(spectrum_to_draw))]
            self.data['display_spectrum{}'.format(int(index))][:len(spectrum_to_draw)] = spectrum_to_draw.real.astype(config.DTYPE)
            self.data['display_spectrum_len{}'.format(int(index))].set(len(spectrum_to_draw))
            
            #spectrum = np.roll(spectrum, int(spectrum.shape[0] * self.p.spectrum_roll()))
            #spectrum = utils.equalize_spectrum(spectrum, self.p.eq_power())
            
            if self.p.innerpad():
                maxfreq = self.p.fmin() + self.p.frange()
            else:
                maxfreq = None
                
            sample = utils.spec2sample(
                spectrum,
                max(self.p.duration(), 4),
                config.SAMPLERATE,
                minfreq=self.p.fmin(),
                maxfreq=maxfreq,
                reso=self.p.reso())
            sample = sample[:int(self.p.duration() * config.SAMPLERATE),:]
            
            sample *= 10**(self.p.volume())
            sample = np.roll(sample, int(sample.shape[0] * self.p.sample_roll()))
            if self.p.loops() > 1:
                sample = np.concatenate(list([sample],) * self.p.loops(), axis=0)
            
            sample = sampler.Sample(sample)
            self.data.set_sample(self.name, sample.sample.astype(config.DTYPE))
            
            
            #self.data.set_sample(self.name, np.copy(Wave(mode='sine').downsample))
            

            
            self.data['display_sample{}'.format(int(index))][:min(config.MAX_DISPLAY_SIZE, len(sample.sample))] = sample.sample[:,0][:min(config.MAX_DISPLAY_SIZE, len(sample.sample))]
            self.data['display_sample_len{}'.format(int(index))].set(min(config.MAX_DISPLAY_SIZE, len(sample.sample)))

            self.data.timing_buffers['synth_computation_time{}'.format(int(index))].put(time.time() - stime)
            self.data['x{}'.format(int(index))].set(self.x)
            self.data['y{}'.format(int(index))].set(self.y)
            time.sleep(config.SYNTH_LOOP_TIME)
            
            


