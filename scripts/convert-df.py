import orb.utils.io as io
import numpy as np
import orb.cube
import orb.utils.image

cube = orb.cube.SpectralCube('/home/thomas/data/M1_2022_SN3.merged.cm1.hdf5')
binning = 5

df = cube.get_deep_frame().data
df = orb.utils.image.nanbin_image(df, binning)

#df = io.read_fits('/home/thomas/music/data/m1.final.deep_frame.wcs.ok.fits')
np.save('m1.sn3.2.df.npy', df)


