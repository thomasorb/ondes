import numpy as np
import pylab as pl
import matplotlib.gridspec
import scipy.fft
import time

from . import core
from . import config

class CubeDisplay(object):

    def __init__(self, data, dfpath):

        assert isinstance(data, core.Data)
        self.data = data
        self.name = 's0'

        GRIDSIZE = 10 #must be even
        SPECSIZE = 3
        PERC = 95
        
        
        self.df = np.load(dfpath).real
        self.x = list()
        self.y = list()
                
        pl.ion()
        self.imfig = pl.figure(figsize=(10,5))
        self.imfig.patch.set_facecolor('black')
        
        gs = matplotlib.gridspec.GridSpec(GRIDSIZE, 2*GRIDSIZE, wspace=1, hspace=0)
        
        self.image_ax = self.imfig.add_subplot(gs[:, 0:GRIDSIZE])
        
        #self.image_ax.cla()
        vmin, vmax = np.nanpercentile(self.df, [30, PERC])
        self.image_ax.imshow(self.df.T, vmin=vmin, vmax=vmax, origin='lower',
                             interpolation='nearest',
                             cmap='hot')
        self.image_ax.axis('off')
        
        (self.xyline,) = self.image_ax.plot([0,0], [0,0], animated=False, color='green', alpha=0.5)
        self.xyscatter = self.image_ax.scatter(0, 0, animated=False, color='green', alpha=0.9, s=20, marker='+')
        
        
        index = 0
        self.sample_ax = self.imfig.add_subplot(gs[index:index+SPECSIZE,GRIDSIZE:])
        index += SPECSIZE
        self.spectrum_ax = self.imfig.add_subplot(gs[index:index+SPECSIZE,GRIDSIZE:])
        index += SPECSIZE
        self.power_ax = self.imfig.add_subplot(gs[index:index+SPECSIZE,GRIDSIZE:])
        
        self.sample = None
        self.spectrum = None
        self.spectrum_ax.plot(np.zeros(100))
        self.sample_ax.plot(np.zeros(100))
        self.power_ax.plot(np.zeros(100))
        
        self.imfig.show()


        self.termfig, self.term_ax = pl.subplots(1,1, figsize=(7,5))
        self.termfig.patch.set_facecolor('black')
        self.term_ax.text(0.35, 0.5, 'Close Me!', dict(size=30))

        pl.pause(0.5)
        self.image_background = self.imfig.canvas.copy_from_bbox(self.image_ax.bbox)
        
        self.image_ax.draw_artist(self.xyline)
        self.image_ax.draw_artist(self.xyscatter)
            
        self.imfig.canvas.blit(self.image_ax.bbox)
        
        
        while True:
            self.spectrum = self.data['display_spectrum'][:self.data['display_spectrum_len'].get()]
            self.sample = self.data['display_sample'][:self.data['display_sample_len'].get()]
            
            self.redraw_plots()
            self.redraw_on_image()
            self.imfig.canvas.flush_events()

            self.redraw_term()
            self.termfig.canvas.flush_events()

            
            
            time.sleep(0.03)
            
            

    def onclick(self, event):
        if event.inaxes not in [self.image_ax]:
            return

        #self.last_spectrum = self.extract_spectrum(int(event.xdata), int(event.ydata))
        
        self.spectrum_ax.cla()
        #compute_effects().play(duration=self.p.duration())
        
        
    def redraw_plot(self, ax, data, title, log=False, xlim=None):
        ax.cla()
        ax.set_facecolor('black')
        ax.plot(data, c='.9', label=title)
        ax.lines[0].set_data(np.arange(np.size(data)), data) # set plot data
        ax.relim()                  # recompute the data limits
        ax.autoscale_view()         # automatic axis scaling
        if xlim is not None:
            ax.set_xlim(xlim)
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        #ax.legend(loc='upper left')
        ax.axis('off')

    def redraw_plots(self):
        if self.spectrum is not None:
            self.redraw_plot(self.spectrum_ax, self.spectrum, 'spectrum')
        if self.sample is not None:
            self.redraw_plot(self.sample_ax, self.sample, 'sample')
            powerspec = np.abs(scipy.fft.fft(self.sample))
            self.redraw_plot(self.power_ax, powerspec[:powerspec.size//2],
                        'power spectrum', xlim=(20, 20000), log=True)
        

        

    def redraw_on_image(self):
        """https://stackoverflow.com/questions/29277080/efficient-matplotlib-redrawing
        """
        new_x = self.data['x'].get() * config.BINNING
        new_y = self.data['y'].get() * config.BINNING
        if len(self.x) == 0 or new_x != self.x[-1] or new_y != self.y[-1]:
            self.x.append(new_x)
            self.y.append(new_y)
            
            #self.image_ax.plot(self.x, self.y, color='tab:green', alpha=0.5)
            #self.image_ax.scatter(self.x[-1], self.y[-1], s=20, color='tab:green', marker='+', alpha=0.9)
        self.imfig.canvas.restore_region(self.image_background)
        self.xyline.set_xdata(self.x)
        self.xyline.set_ydata(self.y)
        self.image_ax.draw_artist(self.xyline)
        self.xyscatter.set_offsets([self.x[-1], self.y[-1]])
        self.image_ax.draw_artist(self.xyscatter)
        self.imfig.canvas.blit(self.image_ax.bbox)
        
            

    def redraw_term(self):

        def get_timing(name):
            _t = self.data.timing_buffers[name].get() * 1000.
            return name + ' {:.2f}|{:.2f}|{:.2f} ms'.format(np.nanmedian(_t), np.nanmax(_t), np.nanmin(_t))
        self.term_ax.cla()
        self.term_ax.set_facecolor('black')
        _s = list()
        _s.append('blocktime (latency) {:.2f} ms'.format(config.BLOCKTIME))
        
        _s.append(get_timing('synth_computation_time'))
        _s.append(get_timing('keyboard_loop_time'))
        _s.append(get_timing('server_callback_time'))
        self.term_ax.text(0., 0., '\n'.join(_s), color='white')
        #self.term_ax.relim()                  # recompute the data limits
        #self.term_ax.autoscale_view()         # automatic axis scaling
        self.term_ax.axis('off')
        
