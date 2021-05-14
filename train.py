import logging
logging.getLogger().setLevel(logging.INFO)
        
import ondes.nn

import soundfile as sf
data, srate = sf.read('../andante.wav')[10000:,:]



brain = ondes.nn.Brain(False)
brain.process(data)
