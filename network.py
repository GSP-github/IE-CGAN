import torch.nn as nn
import torch.nn.parallel
import torch


class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        # 256*256*3 ==>  128*128*64
        self.CR = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias= False),
            nn.ReLU(True)
        )
        # 128*128*64 ==
        self.CBR = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        # 128*128*64 ==> 256*256*3
        self.DT = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias= False),
            nn.Tanh()
        )

    def forward(self, input):
        cr = self.CR(input)
        cbr = self.CBR(cr)
        c = cr + cbr
        out = self.DT(c)
        return out

class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.main = nn.Sequential(
            # 256*256*6 ==> 128*128*64
            nn.Conv2d(6, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # 128*128*64 ==> 64*64*128
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 64*64*128 ==> 32*32*256
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 32*32*256 ==> 29*29*512
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 29*29*512 ==> 30*30*1
            nn.Conv2d(512, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
