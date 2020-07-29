import pulsar.machine
import pulsar.sampler
import numpy as np
import multiprocessing as mp


machine = pulsar.machine.Machine()
sampler = pulsar.sampler.Sampler(machine.data)
machine.play()
