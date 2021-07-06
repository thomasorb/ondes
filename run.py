import logging
logging.getLogger().setLevel(logging.INFO)
        
#import ondes.machine
#cubepath = '../data/crab.npy'
#dfpath = '../data/crab-df.npy'
#machine = ondes.machine.Machine(cubepath, dfpath)

#import ondes.cserver
#ondes.cserver.Server(None)

import ondes.server
ondes.server.Server(None, '/home/thomas/Astro/Interférences/sources/1968/Jodrell Bank (Mitchell Mickaliger)/59329_75673_B0531 21_000000.fil.npy', 15625, invert_channels=True)
