import pulsar.machine
import pulsar.sampler
import pulsar.synth
import pulsar.sequencer
import numpy as np
import multiprocessing as mp
import time

machine = pulsar.machine.Machine()
init = {
    0:'kick0.wav',
    1:'hat0.wav',
    2:'hat1.wav',
}
#sampler = pulsar.sampler.Sampler(machine.data, init=init)
sscore = pulsar.sequencer.SynthScore(machine.data)
sscore[0] << 'C4==|....C#4|....D#4====E4===='.replace('4','2')
#sscore[0] << 'end'

score = pulsar.sequencer.SamplerScore(machine.data)
score[0] << 'end'
score[1] << 'end'
score[2] << 'end'
score[3] << 'end'
synth = pulsar.synth.Synth(1, machine.data, mode='square', dirty=500)
machine.play()
#time.sleep(3)
#machine.pause()
#time.sleep(1)
#synth.play(30, 32)

#sampler.play(2)
#sampler.play(1)
#sampler.play(0)

