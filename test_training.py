import numpy as np
import glob
from ondes import nn
import logging
logging.getLogger().setLevel(logging.INFO)
import soundfile as sf

brain = nn.Brain(False)
in_data, srate = sf.read("../training/Radiohead - Kid A.wav", frames=44100*60, start=44100*60)
if in_data.ndim == 2: 
    in_data = np.mean(in_data, axis=1)
out_data = brain.process(np.copy(in_data), bypass=False)
sf.write('test1.wav', out_data, samplerate=srate)

in_data, srate = sf.read("../training/Cinematic Orchestra - man with a movie camera.wav", frames=44100*60, start=44100*60)
if in_data.ndim == 2: 
    in_data = np.mean(in_data, axis=1)
out_data = brain.process(np.copy(in_data), bypass=False)
sf.write('test2.wav', out_data, samplerate=srate)

out_data_by = brain.process(np.copy(in_data), bypass=True)
sf.write('test_bypass.wav', out_data_by, samplerate=srate)
print(np.std(out_data - out_data_by))

# try morphing

def morph_files(path1, path2, frames=44100*60, start=44100*60):
    in1, srate = sf.read(path1, frames=frames, start=start)
    if in1.ndim == 2: 
        in1 = np.mean(in1, axis=1)
    
    in2, srate = sf.read(path2, frames=frames, start=start)
    if in2.ndim == 2: 
        in2 = np.mean(in2, axis=1)
    
    enc1 = brain.encode(in1.T)
    phase1 = np.copy(brain.data_phase)
    print(brain.data_max)
    
    enc2 = brain.encode(in2.T)
    phase2 = np.copy(brain.data_phase)

    level = np.linspace(0, 1, phase1.shape[0]).reshape((phase1.shape[0],1,1))
    print(level.shape, phase1.shape)
    phase = (1-level) * phase1 + level * phase2
    
    level_enc = np.linspace(1, 0, enc1.shape[0]).reshape((enc1.shape[0],1,1))
    print(level_enc.shape, enc1.shape)
    enc = (1-level_enc) * enc1 + level_enc * enc2
    
    #brain.data_phase = phase
    brain.data_phase = phase
    return brain.decode(enc).T


out_data = morph_files("../training/Cinematic Orchestra - man with a movie camera.wav",
                       "../training/Radiohead - Kid A.wav")
sf.write('test_morph.wav', out_data, samplerate=srate)
