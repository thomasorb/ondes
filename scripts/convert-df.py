import orb.utils.io as io
import numpy as np

df = io.read_fits('/home/thomas/music/data/m1.final.deep_frame.wcs.ok.fits')
np.save('df.npy', df)


