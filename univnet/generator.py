from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import json
import utils

from .lvcnet import LVCBlock

MAX_WAV_VALUE = 32768.0

class Generator(nn.Module):
    """UnivNet Generator"""
    def __init__(self, hp):
        super(Generator, self).__init__()
        self.mel_channel = hp.voc_in_ch  #* 100
        self.noise_dim = hp.voc_noise_dim         #* 64
        self.hop_length = hp.hop_length     #* 256
        channel_size = hp.voc_channel_size      #* 32 or 16 (Univnet c16 or c32)  

        self.res_stack = nn.ModuleList()
        hop_length = 1
        for j, stride in enumerate(hp.strides):               #* [8,8,4]
            hop_length = stride * hop_length        #* 8, 64, 256
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    hp.voc_in_ch,
                    stride=stride,
                    dilations=hp.dilations,
                    lReLU_slope=hp.lReLU_slope,
                    cond_hop_length=hop_length,
                )
            )
                
        #* LVCBlock(16, 100, 8, [1,3,9,27], 0.2, 8)
        #* LVCBlock(16, 100, 8, [1,3,9,27], 0.2, 8)
        #* LVCBlock(16, 100, 4, [1,3,9,27], 0.2, 4)
        
        self.conv_pre = \
            nn.utils.weight_norm(nn.Conv1d(hp.voc_noise_dim, channel_size, 7, padding=3, padding_mode='reflect'))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(hp.lReLU_slope),
            nn.utils.weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )
        

    def forward(self, c, z):
        '''
        Args: 
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length) 
            z (Tensor): the noise sequence (batch, noise_dim, in_length)
        
        '''
        z = self.conv_pre(z)                # (B, c_g, L)

        for res_block in self.res_stack:
            res_block.to(z.device)
            z = res_block(z, c)             # (B, c_g, L * s_0 * ... * s_i)

        z = self.conv_post(z)               # (B, 1, L * 256)

        return z

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        print('Removing weight norm...')

        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def inference(self, c, z=None):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(c.device)
        mel = torch.cat((c, zero), dim=2)
        
        if z is None:
            z = torch.randn(1, self.noise_dim, mel.size(2)).to(mel.device)

        audio = self.forward(mel, z)
        audio = audio.squeeze() # collapse all dimension except time axis
        audio = audio[:-(self.hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()

        return audio

if __name__ == '__main__':
    hp = OmegaConf.load('./default.yaml')
    model = Generator(hp)

    c = torch.randn(3, 100, 10)
    z = torch.randn(3, 64, 10)
    print(c.shape)

    y = model(c, z)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)