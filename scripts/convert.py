import numpy as np

import orb.cube
import orb.utils.image
from orb.core import ProgressBar

cube = orb.cube.SpectralCube('/media/thomas/DATA/local_celeste/Crab-nebula_SN2.merged.cm1.1.0.hdf5')
binning = 4
outpath = 'out.sn2.npy'

zmin, zmax = cube.get_filter_range_pix(border_ratio=0.05).astype(int)



dimx, dimy = orb.utils.image.nanbin_image(np.empty((cube.dimx, cube.dimy)), binning).shape
final_cube = np.empty((dimx, dimy, zmax-zmin), dtype=complex)
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

    
