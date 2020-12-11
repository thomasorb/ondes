import numpy as np
import traceback

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
        self.widget = None
        self.fig = None
        
    def set(self, val):
        self.val = self.cast(np.clip(val, self.vmin, self.vmax))
        
    def toggle(self, event):
        self.val = not bool(self.val)
        if self.val:
            self.widget.ax.set_facecolor('tab:green')
            self.widget.color = 'tab:green'
        else:
            self.widget.ax.set_facecolor('tab:orange')
            self.widget.color = 'tab:orange'
        self.fig.canvas.draw_idle()
            
    def __call__(self):
        return self.cast(self.val)

class Cube(orb.cube.SpectralCube):
    
    def __init__(self, *args, **kwargs):
        """Init class"""
        super().__init__(*args, **kwargs)

        zmin, zmax = self.get_filter_range_pix(border_ratio=0.05).astype(int)
        p = {
            'perc':Param(99., 95., 100., 0.05, cast=float),
            'depth':Param(1, 1, 10, 1),
            'deriv':Param(5, 1, 10, 1),
            'r':Param(10, 1, 30, 1),
            'zmin':Param(zmin, zmin-100, zmin+100, 1),
            'zmax':Param(zmax, zmax-100, zmax+100, 1),
            'reso':Param(2, 0.5, 30, 0.1, cast=float),
            'innerpad':Param(True, False, True, None, cast=bool),
            'fmin':Param(0, 0, 5000, 10, cast=float),
            'frange':Param(500, 200, 5000, 10, cast=float),
            'volume':Param(-2., -4, 1, 0.01, cast=float),
            'duration':Param(3., 1, 30, 0.1, cast=float),
            'loops':Param(1, 1, 10, 1),
            'spectrum_roll':Param(0, -1, 1, 0.01, cast=float),
            'sample_roll':Param(0, -1, 1, 0.01, cast=int),
            'note':Param(64, 0, 127, 1),
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


    def show(self):

        def redraw_plot(ax, data, title, log=False, xlim=None):
            if data.size > 1e6:
                data = data[::int(data.size//1e6),...]
            ax.cla()
            ax.plot(data, c='gray', label=title)
            if xlim is not None:
                ax.set_xlim(xlim)
            if log:
                ax.set_xscale('log')
            ax.legend(loc='upper left')
            ax.axis('off')

        def compute_effects():
            spectrum = np.copy(self.last_spectrum)
            spectrum = np.roll(spectrum, int(spectrum.shape[0] * self.p.spectrum_roll()))
            self.spectrum_to_show = np.copy(utils.normalize_spectrum(spectrum.real, self.p.reso()))
            if self.p.innerpad():
                maxfreq = self.p.fmin() + self.p.frange()
            else:
                maxfreq = None
            sample = utils.spec2sample(
                spectrum,
                self.p.duration(),
                config.SAMPLERATE,
                minfreq=self.p.fmin(),
                maxfreq=maxfreq,
                reso=self.p.reso())
            sample *= 10**(self.p.volume())
            sample = np.roll(sample, int(sample.shape[0] * self.p.sample_roll()))
            if self.p.loops() > 1:
                sample = np.concatenate(list([sample],) * self.p.loops(), axis=0)
            
            sample = sampler.Sample(sample)
            sample = sample.shift(self.p.note() - 64)
            
            
            
            self.last_sample = np.copy(sample.sample)
            redraw_plots()
            return sample

        def onclick(event):
            if event.inaxes not in [self.image_ax]:
                return
            
            self.last_spectrum = self.extract_spectrum(event.xdata, event.ydata)
            
            self.spectrum_ax.cla()
            compute_effects().play(duration=self.p.duration())
            
            
        def redraw_plots():
            redraw_plot(self.spectrum_ax, self.spectrum_to_show, 'spectrum')
            redraw_plot(self.sample_ax, self.last_sample, 'sample')
            powerspec = np.abs(scipy.fft.fft(self.last_sample))
            redraw_plot(self.power_ax, powerspec[:powerspec.size//2],
                        'power spectrum', xlim=(20, 20000), log=True)
            self.imfig.canvas.draw()

        def play_sample(event):
            try:
                compute_effects().play(duration=self.p.duration())
                
            except Exception as e:
                print('error when playing')
                traceback.print_exc(limit=5)
                
    
        def save_sample(event):
            try:
                compute_effects().save(self.path + '.wav')
            except Exception as e:
                print('error when saving')
                traceback.print_exc(limit=5)

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
        
        redraw_plot(self.sample_ax, np.zeros(self.p.zmax() - self.p.zmin()), 'sample')
        redraw_plot(self.spectrum_ax, np.zeros(self.p.zmax() - self.p.zmin()), 'spectrum')
        redraw_plot(self.power_ax, np.zeros(self.p.zmax() - self.p.zmin()), 'power')


        self.parfig = pl.figure(figsize=(4,4))
        gs = matplotlib.gridspec.GridSpec(max(GRIDSIZE, len(self.p) + 4*BUTTSIZE), GRIDSIZE, wspace=1, hspace=0)
        
        index = 0
        for ip in self.p:
            if self.p[ip].cast != bool:
                islider = Slider(self.parfig.add_subplot(gs[index,:]),
                               ip, self.p[ip].vmin,
                               self.p[ip].vmax,
                               valinit=self.p[ip](),
                               valstep=self.p[ip].step)
                islider.on_changed(self.p[ip].set)
                self.p[ip].widget = islider
            else:
                # create button :
                ibutton = Button(self.parfig.add_subplot(gs[index:index+BUTTSIZE,:]), ip)
                ibutton.on_clicked(self.p[ip].toggle)
                self.p[ip].widget = ibutton
            self.p[ip].fig = self.parfig
        
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

    
