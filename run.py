import pulsar.machine
import pulsar.sampler
import numpy as np
import multiprocessing as mp
import time

machine = pulsar.machine.Machine()
sampler = pulsar.sampler.Sampler(machine.data)
synth = pulsar.synth.Synth(1, machine.data, mode='square', dirty=500)
machine.play()
#time.sleep(3)
#machine.pause()
#time.sleep(1)
#synth.play(50, 64)

#sampler.play(2)
#sampler.play(1)
#sampler.play(0)

