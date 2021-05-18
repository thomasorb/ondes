import numpy as np
import time
import datetime
import os

from . import config
import logging

import torch
import torch.nn 
import torch.nn.functional 
import torch.optim

import scipy.fft
import scipy.signal

def hiwei(size, factor=2):
    assert factor >= 1, 'factor must be greater than 1'
    wei = np.arange(size, dtype=float) / size * (factor - 1) + 1
    return wei.reshape((1,1,size))

def normalize(data, threshold):
    data = 20 * np.log10(data) # magnitude is converted to dB (pressure-level)
    # amplify high frequencies to give them more weight for the loss computation
    #data *= hiwei(data.shape[2])
    data[data < threshold] = threshold # data threshold    
    data_min = np.min(data)
    data -= data_min
    data_max = np.max(data)
    data /= data_max / 2.
    data -= 1 # data normalized between -1 and 1
    return data, (data_min, data_max)

def denormalize(data, coeffs):
    # coeffs : data_min, data_max
    data += 1
    data *= coeffs[1] / 2
    data += coeffs[0]
    #data /= hiwei(data.shape[2])
    data = 10**(data / 20)
    return data  


class NeuralNet(torch.nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()

        #DROPOUT_RATE = 0.2
        INCHANNELS = config.NCHANNELS 
        CONV1CHANNELS = INCHANNELS * 4
        CONV2CHANNELS = CONV1CHANNELS * 4

        self.input_size = config.BLOCKSIZE // 2 + 1
        #NFC0 = 257
        NFC1 = 128
        NFC2 = 64
        NFC3 = 16

        def freeze(layer):
            for param in layer.parameters():
                param.requires_grad = False

        ############################################################
        ## WARNING don't forget to set the encoding and decoding layers
        ############################################################
        #self.fc0 = torch.nn.Linear(self.input_size, NFC0)
        self.fc1 = torch.nn.Linear(self.input_size, NFC1)
        self.fc2 = torch.nn.Linear(NFC1, NFC2)
        self.fc3 = torch.nn.Linear(NFC2, NFC3)
        self.fc3out = torch.nn.Linear(NFC3, NFC2)
        self.fc2out = torch.nn.Linear(NFC2, NFC1)
        self.fc1out = torch.nn.Linear(NFC1, self.input_size)
        #self.fc0out = torch.nn.Linear(NFC0, self.input_size)
        
        #self.drop = torch.nn.Dropout(DROPOUT_RATE)
        
        self.reconstruction_loss = torch.nn.MSELoss()

        
    def forward(self, x):
        # Autoencoders for music sound modeling: a comparison of
        # linear, shallow, deep,recurrent and variational models
        # https://arxiv.org/abs/1806.04096
        
        nbatch = x.shape[0]
        x = self.encode(x)
        x = self.decode(x)
        
        self.latent_loss = 0
        
        return x.reshape((nbatch, config.NCHANNELS, self.input_size))

    def encode(self, x):
        #x = torch.tanh(self.fc0(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
    
    def decode(self, x):
        x = torch.tanh(self.fc3out(x))
        x = torch.tanh(self.fc2out(x))
        x = self.fc1out(x)
        #x = self.fc0out(x)
        return x

    
class Brain(object):

    MAX_BATCH_SIZE = 0.3 # Gb
        
    def __init__(self, train, lr=0.002):

        self.train = train
        if self.train:
            self.index = 0
        
        self.net = NeuralNet()
        try:
            self.net.load_state_dict(torch.load(config.NNPATH), strict=True)
        except Exception as e:
            logging.warning('exception during load state:', e)
        else:
            self.net.eval()
            
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.optimizer.zero_grad()

    def data_is_valid(self, data):
        #assert data.shape[0] == config.NCHANNELS, 'data must have shape ({},n) but has shape {}'.format(config.NCHANNELS, data.shape)
        assert data.ndim == 1
        
    def process_large_data(self, data, **kwargs):
        assert self.train, 'must be in training mode'
        nchunks = int(data.size * data.itemsize / 1e9 / (0.5 * self.MAX_BATCH_SIZE)) + 1
        chunks = np.linspace(0, data.size, nchunks).astype(int)
        losses = list()
        for i in range(len(chunks)-1):
            try: del out
            except: pass
            logging.info('processing chunk {}/{}'.format(i+1, nchunks))
            out, loss = self.process(data[chunks[i]:chunks[i+1]], **kwargs)
            losses.append(loss)
        
        return out, losses
            
    def preprocess(self, data, bypass=False):
        
        DB_THRESHOLD = -100 # dB
        DATA_NOISE_COEFF = 0.0001

        self.data_is_valid(data)

        
        nbatch = data.size // config.BLOCKSIZE

        
        if nbatch * config.BLOCKSIZE != data.size and not self.train:
            logging.warning('number of samples is not a multiple of blocksize')
        
        data = np.copy(data[:nbatch * config.BLOCKSIZE])

        # noise is added
        if self.train and not bypass:
            data += np.random.standard_normal(size=data.size).reshape(data.shape) * DATA_NOISE_COEFF * np.max(data)
            
        #data = data.reshape((nbatch, config.NCHANNELS, config.BLOCKSIZE))
        #data_fft = scipy.fft.rfft(data)[:,:,:-1]
        
        _, _, data_fft = scipy.signal.stft(data, nperseg=config.BLOCKSIZE)
        print(data_fft.shape)
        
        data_fft = data_fft.reshape((data_fft.shape[-1], 1 ,data_fft.shape[0]))
        self.data_fft_shape = data_fft.shape
        self.data_phase = np.angle(data_fft) # phase detached and kept to the end
        data = np.abs(data_fft) # only magnitude goes through the network
        
        data, self.data_max = normalize(data, DB_THRESHOLD)
        
        del data_fft
        return data
    
    def postprocess(self, data):
        data = data.reshape(self.data_fft_shape)

        # data nomalization removal
        data = denormalize(data, self.data_max)
        data_fft = np.empty(self.data_fft_shape, dtype=complex)
        data_fft.real = data * np.cos(self.data_phase)
        data_fft.imag = data * np.sin(self.data_phase)
        #data = scipy.fft.irfft(data_fft, n=config.BLOCKSIZE).astype(np.float32)
        data_fft = data_fft.reshape((data_fft.shape[-1], data_fft.shape[0]))
        _, data = scipy.signal.istft(data_fft, nfft=config.BLOCKSIZE, nperseg=config.BLOCKSIZE)
        data = data.reshape((data.size)).astype(np.float32)
        del data_fft
        del self.data_phase
        return data
    
    def process(self, data, bypass=False):

        #RECON_NOISE_COEFF = 0.000001

        self.data_is_valid(data)
        
        if (data.size * data.itemsize) / 1e9 > self.MAX_BATCH_SIZE:
            if self.train:
                logging.warning('data is too large and will be treated in smaller chunk, only the last chunk will be returned')
                return self.process_large_data(data, bypass=bypass)
            else:
                raise Exception('data file too large for pure processing (without training)')

        data = self.preprocess(data, bypass=bypass)
        
        if not bypass:
            
            # if self.train:
            #     # add noise also to the data passed to the network
            #     rec_noise = np.random.standard_normal(size=data.size).reshape(data.shape) * RECON_NOISE_COEFF
            #     data = torch.from_numpy(data)
            #     data_out = self.net(data.float() + torch.from_numpy(rec_noise).float()).double()
            # else:
            #     data = torch.from_numpy(data)
            #     data_out = self.net(data.float()).double()
            data = torch.from_numpy(data)
            data_out = self.net(data.float()).double()

        else:
            data_out = np.copy(data)
            
        if self.train and not bypass:
            rec_loss = self.net.reconstruction_loss(data_out, data)
            loss = rec_loss + self.net.latent_loss
            logging.info('({}) losses: {}, {}'.format(self.index, rec_loss.item(), self.net.latent_loss))
            loss.backward() # compute gradient

            self.optimizer.step()
            self.optimizer.zero_grad()

        if not bypass:
            data = data_out.detach().numpy()
        del data_out

        data = self.postprocess(data)
            
        # if self.train:
        #     self.index += 1
        #     if not self.index%4 and self.index > 0:
        #         try:
        #             logging.debug('saving')
        #             torch.save(self.net.state_dict(), config.NNPATH + '.{}'.format(datetime.datetime.timestamp(datetime.datetime.now())))
        #         except Exception as e:
        #             logging.warning('exception during model save', e)

        if self.train:
            return data, loss.item()
        else:
            return data


    def encode(self, data):
        self.data_is_valid(data)
        data = self.preprocess(data)
        data = torch.from_numpy(data)
        return self.net.encode(data.float()).double().detach().numpy()


    def decode(self, data):
        data = torch.from_numpy(data)
        data = self.net.decode(data.float()).double().detach().numpy()
        return self.postprocess(data)
        

    def save(self):
        try:
            torch.save(self.net.state_dict(), config.NNPATH)
        except Exception as e:
            logging.warning('exception during model save', e)
        
    def tempsave(self):
        try:
            torch.save(self.net.state_dict(), config.NNPATH + '.{}'.format(datetime.datetime.timestamp(datetime.datetime.now())))
        except Exception as e:
            logging.warning('exception during model save', e)
            
    # def __del__(self):
    #     try:
    #         if self.train:
    #             torch.save(self.net.state_dict(), config.NNPATH)
    #     except Exception as e:
    #         logging.warning('exception during model save', e)
            
        
