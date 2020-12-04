import numpy as np

from . import utils
from . import config
from . import effects
from . import ccore
from . import sampler
from . import maths

import orb.cube
import orb.core

import pylab as pl
import matplotlib.gridspec
import scipy.interpolate
import sounddevice as sd

from matplotlib.widgets import Slider, Button, TextBox

class Param(object):

    def __init__(self, val, vmin, vmax, step, cast=int):
        self.cast = cast
        self.vmin = self.cast(vmin)
        self.vmax = self.cast(vmax)
        self.val = self.cast(val)
        self.step = self.cast(step)

    def set(self, val):
        self.val = self.cast(np.clip(val, self.vmin, self.vmax))
        

    def __call__(self):
        return self.cast(self.val)

class Cube(orb.cube.SpectralCube):
    
    def __init__(self, *args, **kwargs):
        """Init class"""
        super().__init__(*args, **kwargs)

        zmin, zmax = self.get_filter_range_pix(border_ratio=0.05).astype(int)
        zmin, zmax = 500, 1200 ## HACK !!!!
        p = {
            'depth':Param(1, 1, 10, 1),
            'loops':Param(1, 1, 10, 1),
            'deriv':Param(5, 1, 10, 1),
            'r':Param(10, 1, 30, 1),
            'zmin':Param(zmin, zmin-100, zmin+100, 1),
            'zmax':Param(zmax, zmax-100, zmax+100, 1),
            'note':Param(0, 0, 127, 1),
            'perc':Param(99., 95., 100., 0.05, cast=float),
            'reso':Param(2, 0.5, 30, 0.1, cast=float),
            'fmin':Param(0, 0, 5000, 10, cast=float),
            'fmax':Param(300, 200, 5000, 10, cast=float),
            'volume':Param(-1., -2, 1, 0.01, cast=float),
            'duration':Param(3., 1, 30, 0.1, cast=float),
        }
        self.p = orb.core.Params(p)
        
        self.base_norm_r = 1.
        self.base_norm_i = 1.
        self.ref_spectrum = self.extract_spectrum(
            self.dimx//2, self.dimy//2)
        self.base_norm_r = np.nanpercentile(self.ref_spectrum.real, 99.9)
        self.base_norm_i = np.nanpercentile(self.ref_spectrum.imag, 99.9)
        self.last_spectrum = None
        self.last_sample = None
        
        
        
    # def get_sample(self, ix, iy, asarr=False):
    #     rx, ry = np.random.randint(-self.p.deriv(), self.p.deriv(), 2)
        
    #     sampleL = self.extract_sample(ix, iy)
    #     sampleR = self.extract_sample(ix, iy)
    #     sample = np.array((sampleL, sampleR)).T
    #     sample = effects.cut_to_blocksize(sample, config.BLOCKSIZE)
    #     if asarr:
    #         return sample
    #     else:
    #         return sampler.Sample(sample)
    

    # def get_samples(self, ix, iy, asarr=False):
    #     dataLR = list()
    #     for iloop in range(self.p.loops()):
    #         spec = self.get_sample(ix, iy, asarr=True)
    #         dataLR.append(spec)
            
    #     dataLR = np.concatenate(dataLR)
    #     dataLR = effects.cut_to_blocksize(dataLR, config.BLOCKSIZE)

    #     if asarr:
    #         return dataLR
    #     else:
    #         return sampler.Sample(dataLR)
            
    def extract_spectrum(self, ix, iy):
        ix = int(ix)
        iy = int(iy)
        data = list()
        r = int(self.p.r())
        for iloop in range(self.p.depth()):
            idata = self[ix-r:ix+r+1, iy-r:iy+r+1, self.p.zmin():self.p.zmax()]
            idata = np.mean(idata, axis=(0,1))
            #idata.real /= self.base_norm_r
            #idata.imag /= self.base_norm_r
            #if iloop&1:
            #    data.append(idata[::-1])
            #else:
            data.append(idata)
            
            rx, ry = np.random.randint(-self.p.deriv(), self.p.deriv(), 2)
            ix += rx
            iy += ry
        data = np.concatenate(data)
        return data

    # def extract_sample(self, ix, iy):
    #     def special_treatment(interf):
    #         interf -= np.mean(interf)

    #         interf /= np.nanpercentile(interf, 99.95)
    #         interf = np.clip(interf, -1, 1)
    #         interf /= 2.

    #         #interf = np.concatenate((interf[interf.size//2:], interf[:interf.size//2]))
    #         interf = interf[interf.size//5:interf.size//3]
    #         return interf

    #     spectrum = self.extract_spectrum(ix, iy)
    #     self.last_spectrum = np.copy(spectrum)
    #     sample = utils.inverse_transform(spectrum, 10, config.BASENOTE, config.A_MIDIKEY).real
    #     sample = special_treatment(sample)
    #     self.last_sample = np.copy(sample)
    #     return sample


    def show(self):

        def redraw_plot(ax, data, title, log=False, xlim=None):
            ax.cla()
            ax.plot(data, c='gray', label=title)
            if xlim is not None:
                ax.set_xlim(xlim)
            if log:
                ax.set_xscale('log')
            ax.legend()
            ax.axis('off')

        def compute_effects():
            print(self.p.fmin())
            print(self.p.fmax())
            print(self.last_spectrum.shape)
            sample = utils.spec2sample(
                self.last_spectrum * 10**(self.p.volume()),
                self.p.duration(),
                config.SAMPLERATE,
                minfreq=self.p.fmin(),
                maxfreq=self.p.fmax(),
                reso=self.p.reso())
            self.last_sample = np.copy(sample)
            
            return sampler.Sample(sample)

        def onclick(event):
            if event.inaxes not in [self.image_ax]:
                return
            
            self.last_spectrum = self.extract_spectrum(event.xdata, event.ydata)
            
            self.spectrum_ax.cla()
            compute_effects().play(duration=5)

            redraw_plot(self.spectrum_ax, self.last_spectrum.real, 'spectrum')
            np.save('last_spectrum.npy', self.last_spectrum)
            redraw_plot(self.sample_ax, self.last_sample, 'sample')

            powerspec = np.abs(scipy.fft.fft(self.last_sample))
            #freq = scipy.fft.fftfreq(self.last_sample.size, 1/config.SAMPLERATE)
            #pl.plot(freq[:fft.size//2], )
            #pl.xlim
            #pl.xscale('log')
            redraw_plot(self.power_ax, powerspec[:powerspec.size//2], 'sample', xlim=(20, 20000), log=True)
            self.imfig.canvas.draw()

        def play_sample(event):
            try:
                compute_effects().play()
                
            except Exception as e:
                print(e)
    
        def save_sample(event):
            try:
                compute_effects().save(self.path + '.wav')
            except Exception as e:
                print(e)

        def change_path(path):
            self.path = str(path)

        def redraw_image(event):
            self.image_ax.cla()
            vmin, vmax = np.nanpercentile(df, [1,self.p.perc()])
            self.image_ax.imshow(df.T, vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')
            self.image_ax.axis('off')
            cid = self.imfig.canvas.mpl_connect('button_press_event', onclick)
            self.imfig.canvas.draw()
                        
        GRIDSIZE = 10 #must be even
        SPECSIZE = 1
        BUTTSIZE = 1

        
        
        df = self.get_deep_frame().data
        self.imfig = pl.figure(figsize=(8,10))

        gs = matplotlib.gridspec.GridSpec(GRIDSIZE + 3*SPECSIZE, GRIDSIZE, wspace=1, hspace=0)
        
        self.image_ax = self.imfig.add_subplot(gs[0:GRIDSIZE, :])
        redraw_image(None)
        index = GRIDSIZE
        self.sample_ax = self.imfig.add_subplot(gs[index:index+SPECSIZE,:])
        index += SPECSIZE
        self.spectrum_ax = self.imfig.add_subplot(gs[index:index+SPECSIZE,:])
        index += SPECSIZE
        self.power_ax = self.imfig.add_subplot(gs[index:index+SPECSIZE,:])
        
        redraw_plot(self.sample_ax, np.arange(self.p.zmin(), self.p.zmax()), 'sample')
        redraw_plot(self.spectrum_ax, np.arange(self.p.zmin(), self.p.zmax()), 'spectrum')
        redraw_plot(self.power_ax, np.arange(self.p.zmin(), self.p.zmax()), 'power')


        self.parfig = pl.figure(figsize=(4,4))
        gs = matplotlib.gridspec.GridSpec(max(GRIDSIZE, len(self.p) + 4*BUTTSIZE), GRIDSIZE, wspace=1, hspace=0)
        
        index = 0
        self.sliders = list()
        for ip in self.p:
            islider = Slider(self.parfig.add_subplot(gs[index,:]),
                           ip, self.p[ip].vmin,
                           self.p[ip].vmax,
                           valinit=self.p[ip](),
                           valstep=self.p[ip].step)
            islider.on_changed(self.p[ip].set)
            self.sliders.append(islider)
            index += 1

        self.button_play = Button(self.parfig.add_subplot(gs[index:index+BUTTSIZE,:]),
                                  'play')
        self.button_play.on_clicked(play_sample)
        index += BUTTSIZE

        self.path = 'temp'
        self.path_box = TextBox(self.parfig.add_subplot(gs[index:index+BUTTSIZE,:]),
                                '', initial=self.path)
        self.path_box.on_submit(change_path)
        index += BUTTSIZE

        self.button_save = Button(self.parfig.add_subplot(gs[index:index+BUTTSIZE,:]),
                                  'save')
        self.button_save.on_clicked(save_sample)
        index += BUTTSIZE

        self.button_redraw = Button(self.parfig.add_subplot(gs[index:index+BUTTSIZE,:]),
                                  'redraw')
        self.button_redraw.on_clicked(redraw_image)
        index += BUTTSIZE

        
        pl.show()

    
