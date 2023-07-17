import torch
import torch.nn as nn
import torch.nn.functional as F
import os,sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_dir.split('/')[:-2]))
from utils.model_parts2 import DoubleConv, Down, SelfAttention, Up, one_param













class UNet_conditional(nn.Module):
    def __init__(self, num_channels=28, input_dim = 27,c_out=28, kernel_size=3,time_dim=128,device='cpu'):
        super().__init__()
        print(kernel_size)
        # self.encInput =nn.Sequential(
        # nn.Conv1d(27,28,constants.kernel_size,padding="same",bias=True),
        # nn.ReLU(),
        # nn.Conv1d(28,128,constants.kernel_size,padding="same",bias=True),
        # nn.ReLU(),
        # )

        self.device = device
        self.time_dim = time_dim
        self.encoding = nn.Linear(100,128)
        self.encodingc = nn.Linear(100,128)
        self.inc = nn.Conv1d(num_channels,128,kernel_size,padding="same")
        self.inc2 = DoubleConv(128, 128, kernel_size)
        self.down1 = Down(128, 256, kernel_size)
        self.p1 = nn.Linear(input_dim,256)
        self.sa1 = SelfAttention(256)
        self.down2 = Down(256, 512, kernel_size)
        self.sa2 = SelfAttention(512)
        self.p2 = nn.Linear(input_dim, 512)
        self.down3 = Down(512, 512, kernel_size)
        self.sa3 = SelfAttention(512)
        self.p3 = nn.Linear(input_dim, 512)

        # self.bot1 = DoubleConv(256, 512)
        # self.bot2 = DoubleConv(512, 512)
        # self.bot3 = DoubleConv(512, 256)

        #self.bot1 = DoubleConv(512, 512,kernel_size)
        # self.bot2 = DoubleConv(512, )
        #self.bot2 = DoubleConv(512, 512, constants.kernel_size)
        #self.bot3 = DoubleConv(512, 512,kernel_size)
        self.mid = nn.Sequential(DoubleConv(512,512,kernel_size),DoubleConv(512,512,kernel_size))

        self.up1 = Up(1024, 512,kernel_size)
        #self.up1 = Up(1024, 512, constants.kernel_size)

        self.sa4 = SelfAttention(512)
        self.p4 = nn.Linear(input_dim, 512)
        self.up2 = Up(768, 256,kernel_size)
        #self.up2 = Up(768, 256, constants.kernel_size)
        self.sa5 = SelfAttention(256)
        self.p5 = nn.Linear(input_dim, 256)
        self.up3 = Up(384, 128,kernel_size)
        #self.up3 = Up(384, 128, constants.kernel_size)
        self.sa6 = SelfAttention(128)
        self.p6 = nn.Linear(input_dim, 128)
        self.lastResNet = nn.Sequential(DoubleConv(128,128,kernel_size),DoubleConv(128,128,kernel_size))
        self.outc = nn.Conv1d(128,c_out, kernel_size=1)
        #self.outc = nn.Conv1d(128, 100, kernel_size=1)
        self.decoding = nn.Linear(128,100)


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t, c):
        c = c.to(self.device)
        c = self.encodingc(c).swapaxes(1,2).float().to(self.device)
    
        #c = c.reshape(-1,c.shape[2],c.shape[1])
        #print(c.shape)
        x = x.to(self.device)
        
        x = self.encoding(x)
        x = self.inc(x)
        x1 = self.inc2(x)
        
        x1 =(x1 + t[:,:,None].repeat(1,1,128))
        x2 = self.down1(x1)
        
        c1 = self.p1(c)
        x2 = self.sa1(x2,c1)
        x3 = self.down2(x2)
        
        c2 = c.swapaxes(1, 2)
        c2 = self.p2(c) 
        x3 = self.sa2(x3,c2)
        x4 = self.down3(x3)


        c3 = self.p3(c)
        x4 = self.sa3(x4,c3)

        x4 = self.mid(x4)
        #x = self.up1(x5, x3)
        x = self.up1(x4, x3)


        c4 = self.p4(c)
        x = self.sa4(x,c4)

       
        x = self.up2(x, x2)
        
        c5 = self.p5(c)
        x = self.sa5(x,c5)
        x = self.up3(x, x1)
        c6 = self.p6(c)
        x = self.sa6(x,c6)
        x = self.lastResNet(x)
        output = self.outc(x)
        output = self.decoding(output)
        return output



    def forward(self, x, t,c):
        t = t.unsqueeze(-1).to(self.device)
        t = self.pos_encoding(t, self.time_dim)

        return self.unet_forwad(x, t,c)

if __name__ == "__main__":
    A = torch.randn(32,28,100)
    B = torch.randn(32,27,100)
    C = torch.randn(32)
    model = UNet_conditional()
    model(A,C,B)
