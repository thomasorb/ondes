import numpy as np

import orb.cube
import orb.utils.image
from orb.core import ProgressBar

cube = orb.cube.SpectralCube('/home/thomas/data/M1_2022_SN3.merged.cm1.hdf5')
binning = 5
outpath = 'crab.sn3.2.npy'
dtype = float

zmin, zmax = cube.get_filter_range_pix(border_ratio=0.05).astype(int)



dimx, dimy = orb.utils.image.nanbin_image(np.empty((cube.dimx, cube.dimy)), binning).shape
final_cube = np.empty((dimx, dimy, zmax-zmin), dtype=dtype)
print('out cube shape:', final_cube.shape)
print('out cube size:', final_cube.size * final_cube.itemsize / 1e9, 'Gb')

progress = ProgressBar(zmax - zmin)
for iz in range(zmin, zmax):
    progress.update(iz-zmin)
    iframe = cube[:,:,iz]
    final_cube[:,:,iz-zmin] = orb.utils.image.nanbin_image(iframe, binning)
progress.end()

print('saving')
np.save(outpath, final_cube)

    
