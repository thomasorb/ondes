if __name__ == '__main__':
    import logging
    import platform
    import multiprocessing

    import sounddevice
    print(sounddevice.query_devices())

    #devices = mido.get_output_names()
    #print('midi devices:\n{}'.format('\n  '.join(devices)))

    #cubepath = '../data/crab.sn2.npy'
    #dfpath = '../data/crab-df.npy'
    #machine = ondes.machine.Machine(cubepath, 0.3, dfpath)
    
    print(platform.system())
    #if platform.system() == "Darwin":
    multiprocessing.set_start_method('spawn')
    
    import ondes.machine
    #cubepath = '../data/crab.npy'
    #machine = ondes.machine.Machine(cubepath, dfpat
    #phasepath = '../data/crab2d_phase.npy'
    
    
    cubepath = '../data/crab.npy'
    dfpath = '../data/crab-df.npy'

    machine = ondes.machine.Machine(cubepath, 0.3, dfpath)
    #machine = ondes.machine.Machine('../data/59329_75673_B0531 21_000000.fil.npy', 15625, None)
    #machine = ondes.machine.Machine('../data/59329_75073_B0329 54_000002.fil.npy', 9500)
    #machine = ondes.machine.Machine('../data/59329_37387_B1933 16_000002.fil.npy', 9500)

