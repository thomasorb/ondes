import logging
logging.getLogger().setLevel(logging.INFO)
        
import ondes.machine
cubepath = '../data/crab.npy'
dfpath = '../data/crab-df.npy'
machine = ondes.machine.Machine(cubepath, dfpath)
