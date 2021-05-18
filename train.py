import numpy as np
import glob
from ondes import nn
import logging
logging.getLogger().setLevel(logging.INFO)
import soundfile as sf

files = glob.glob("../training/*.wav")
if len(files) == 0:
    print('no files found')
    quit()
    
brain = nn.Brain(True)
all_losses = list()
stop = False
for iepoch in range(1000):
    if stop: break
    losses = list()
    print('=====================================================================')
    print('                      EPOCH: ', iepoch + 1)
    print('=====================================================================')

    for ifile in files:
        if stop: break
        try:
            print('loading', ifile)
            in_data, srate = sf.read(ifile)
            if in_data.ndim == 2: 
                in_data = np.mean(in_data, axis=1)
            _, ilosses = brain.process(in_data, bypass=False)
            try: len(ilosses)
            except: losses.append(ilosses)
            else: losses += ilosses
            brain.tempsave()
            brain.save()
            np.save('all_losses.npy', all_losses)
            del in_data
            print('losses up to now:', np.mean(losses), np.std(losses))

        except Exception as e:
            print('exception occured during training: ', e)
            stop = True
            break



    all_losses.append(np.mean(losses))

