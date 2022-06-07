import numpy as np
#import pylab as pl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import matplotlib.gridspec
import scipy.fft
import time

from . import core
from . import config
from . import utils

class SynthData(object):

    def __init__(self, index):
        self.index = int(index)
        self.x = list()
        self.y = list()
        self.xyline = None
        self.xyscatter = None
        self.spectrum = None
        #self.sample = None
        
    def add_xy(self, new_x, new_y):
        if len(self.x) == 0 or new_x != self.x[-1] or new_y != self.y[-1]:
            self.x.append(new_x)
            self.y.append(new_y)

        if self.xyline is not None:
            self.xyline.set_xdata(self.x)
            self.xyline.set_ydata(self.y)

        if self.xyscatter is not None:
            self.xyscatter.set_offsets(
                [self.x[-1], self.y[-1]])
            
        
class CubeDisplay(object):

    colors = ['#afff76', '#7fc5ff', '#B682FF']    

    def __init__(self, data, dfpath):

        assert isinstance(data, core.Data)
        self.data = data
        
        GRIDSIZE = 10 #must be even
        SPECSIZE = 3
        PERC = 99
        
        self.df = np.load(dfpath).real
        self.synths = list()
        for i in range(config.MAX_SYNTHS):
            self.synths.append(SynthData(i))
            
        pl.ion()
        self.imfig = pl.figure(figsize=(10,5))
        self.imfig.patch.set_facecolor('black')
        self.imfig.canvas.mpl_connect('button_press_event', self.onclick)
        
        gs = matplotlib.gridspec.GridSpec(GRIDSIZE, 2*GRIDSIZE, wspace=1, hspace=0)
        
        self.image_ax = self.imfig.add_subplot(gs[:, 0:GRIDSIZE])
        
        #self.image_ax.cla()
        vmin, vmax = np.nanpercentile(self.df, [30, PERC])
        self.image_ax.imshow(self.df.T, vmin=vmin, vmax=vmax, origin='lower',
                             interpolation='nearest',
                             cmap='hot')
        self.image_ax.axis('off')

        for i in range(config.MAX_SYNTHS):
        
            (self.synths[i].xyline, ) = self.image_ax.plot(
                [0,0], [0,0], animated=False, color=self.colors[i], alpha=0.5)
            self.synths[i].xyscatter = self.image_ax.scatter(
                0, 0, animated=False, color=self.colors[i], alpha=1, s=30, marker='+')
        
        
        index = 0
        #self.sample_ax = self.imfig.add_subplot(gs[index:index+SPECSIZE,GRIDSIZE:])
        #index += SPECSIZE
        self.spectrum_ax = self.imfig.add_subplot(gs[index:index+SPECSIZE,GRIDSIZE:])
        index += SPECSIZE
        #self.power_ax = self.imfig.add_subplot(gs[index:index+SPECSIZE,GRIDSIZE:])
        
        self.spectrum_ax.plot(np.zeros(100))
        #self.sample_ax.plot(np.zeros(100))
        #self.power_ax.plot(np.zeros(100))
        
        self.imfig.show()


        self.termfig, self.term_ax = pl.subplots(1,1, figsize=(7,5))
        self.termfig.patch.set_facecolor('black')
        self.term_ax.text(0.35, 0.5, '', dict(size=30))

        pl.pause(0.5)
        self.image_background = self.imfig.canvas.copy_from_bbox(self.image_ax.bbox)

        for i in range(config.MAX_SYNTHS):
            self.image_ax.draw_artist(self.synths[i].xyline)
            self.image_ax.draw_artist(self.synths[i].xyscatter)
            
        self.imfig.canvas.blit(self.image_ax.bbox)
        
        
        while True:
            for i in range(config.MAX_SYNTHS):        
                self.synths[i].spectrum = self.data['display_spectrum{}'.format(i)][:self.data['display_spectrum_len{}'.format(i)].get()]
                #self.synths[i].sample = self.data['display_sample{}'.format(i)][:self.data['display_sample_len{}'.format(i)].get()]
                
            self.redraw_plots()
            self.redraw_on_image()
            self.imfig.canvas.flush_events()

            self.redraw_term()
            self.termfig.canvas.flush_events()
            
            time.sleep(0.1)
            
            

    def onclick(self, event):
        if event.inaxes not in [self.image_ax]:
            return
        self.data['x_orig0'].set(int(event.xdata // config.BINNING))
        self.data['y_orig0'].set(int(event.ydata // config.BINNING))
        
    def redraw_plot(self, ax, data, title, log=False, xlim=None, axis=None):
        
        ax.cla()
        ax.set_facecolor('black')
        for i in range(len(data)):
            if axis is None:
                ax.plot(data[i], c=self.colors[i], label=title, alpha=0.5)
            else:
                ax.plot(axis[i], data[i], c=self.colors[i], label=title, alpha=0.5)
            ax.lines[i].set_data(np.arange(np.size(data[i])), data[i]) # set plot data
        ax.relim()                  # recompute the data limits
        ax.autoscale_view()         # automatic axis scaling
        if xlim is not None:
            ax.set_xlim(xlim)
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.axis('on')
            ax.spines['bottom'].set_color('white')
            ax.xaxis.set_ticks([20, 200, 2000, 20000])
            ax.tick_params(axis='x', colors='white')
            
            ax.set_xticklabels([20, 200, 2000, 20000])

        #ax.legend(loc='upper left')
        
    def redraw_plots(self):
        spectra = list()
        #samples = list()
        #powersp = list()
        #powaxes = list()
        for i in range(config.MAX_SYNTHS):
            if self.synths[i].spectrum is not None:
                spectra.append(self.synths[i].spectrum)
            #if self.synths[i].sample is not None:
            #    samples.append(self.synths[i].sample)
            #    iaxis, ipow = utils.power_spectrum(self.synths[i].sample, config.SAMPLERATE)
            #    powersp.append(ipow)
            #    powaxes.append(iaxis)
                
        if len(spectra) > 0:
            self.redraw_plot(self.spectrum_ax, spectra, 'spectrum')
        # if len(samples) > 0:
        #     self.redraw_plot(self.sample_ax, samples, 'sample')
        #     self.redraw_plot(self.power_ax, powersp, 'power spectrum', xlim=(20, 20000), log=True, axis=powaxes)

                
        

    def redraw_on_image(self):
        """https://stackoverflow.com/questions/29277080/efficient-matplotlib-redrawing
        """
        self.imfig.canvas.restore_region(self.image_background)
        
        for i in range(config.MAX_SYNTHS):
            new_x = self.data['x{}'.format(i)].get() * config.BINNING
            new_y = self.data['y{}'.format(i)].get() * config.BINNING
            self.synths[i].add_xy(new_x, new_y)
            
            self.image_ax.draw_artist(self.synths[i].xyline)
            self.image_ax.draw_artist(self.synths[i].xyscatter)
            
        self.imfig.canvas.blit(self.image_ax.bbox)
        
            

    def redraw_term(self):

        def get_timing(name):
            _t = self.data.timing_buffers[name].get() * 1000.
            if not np.all(np.isnan(_t)):
                return name + ' {:.2f}|{:.2f}|{:.2f} ms'.format(
                    np.nanmedian(_t), np.nanmax(_t), np.nanmin(_t))
            else:
                return name
            
        self.term_ax.cla()
        self.term_ax.set_facecolor('black')
        _s = list()
        _s.append('blocktime (latency) {:.2f} ms'.format(config.BLOCKTIME))

        #for i in range(config.MAX_SYNTHS):
        #    _s.append(get_timing('synth_computation_time{}'.format(i)))
        _s.append(get_timing('midi_loop_time'))
        #_s.append(get_timing('keyboard_next_block_time'))
        _s.append(get_timing('server_callback_time'))
        _s.append('xy {} {}'.format(
            self.data['x0'].get(),
            self.data['y0'].get()))
        self.term_ax.text(0., 0., '\n'.join(_s), color='white')
        #self.term_ax.relim()                  # recompute the data limits
        #self.term_ax.autoscale_view()         # automatic axis scaling
        self.term_ax.axis('off')
        
