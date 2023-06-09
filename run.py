import logging
logging.getLogger().setLevel(logging.INFO)

import sounddevice as sd
devices = sd.query_devices()
logging.info('audio devices: \n{}'.format(repr(devices)))
                
import ondes.machine
#cubepath = '../data/crab.npy'
#machine = ondes.machine.Machine(cubepath, dfpat
#phasepath = '../data/crab2d_phase.npy'


cubepath = '../data/crab.sn3.2.npy'
dfpath = '../data/crab-df.npy'
machine = ondes.machine.Machine(cubepath, 0.3, dfpath)



#machine = ondes.machine.Machine('/home/thomas/Astro/Interférences/sources/1968/Jodrell Bank (Mitchell Mickaliger)/59329_75673_B0531 21_000000.fil.npy', 15625, None)
##machine = ondes.machine.Machine('/home/thomas/Astro/Interférences/sources/1968/Jodrell Bank (Mitchell Mickaliger)/59329_75073_B0329 54_000002.fil.npy', 9500)
##machine = ondes.machine.Machine('/home/thomas/Astro/Interférences/sources/1968/Jodrell Bank (Mitchell Mickaliger)/59329_37387_B1933 16_000002.fil.npy', 9500)


