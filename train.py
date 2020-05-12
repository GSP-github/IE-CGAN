import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import numpy as np

import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
import time

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from network import NetG,NetD
from data import GetMateData

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def load_data(dataroot):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    root1 = os.path.join(dataroot, 'high')
    root2 = os.path.join(dataroot, 'low')
    dataset = GetMateData(root1, root2, transform, transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=int(2),
                                                 drop_last=True)
    return dataloader


def train():

    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    netg = NetG().to(device)
    netg.apply(weights_init)
    netd = NetD().to(device)
    netd.apply(weights_init)
    print(netd, netg)

    criterion = nn.MSELoss()
    optimizerD = optim.Adam(netd.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netg.parameters(), lr=0.0002, betas=(0.5, 0.999))

    real_lable = torch.tensor(np.ones((1,1,30,30))).float()
    fake_lable = torch.tensor(np.zeros((1,1,30,30))).float()

    # real_lable = torch.tensor(np.ones((1,1,30,30))).float().cuda()
    # fake_lable = torch.tensor(np.zeros((1,1,30,30))).float().cuda()

    # LOAD DATA
    dataloader = load_data('F:/CGAN/DATA')

    #losses
    G_LOSS = []
    D_LOSS = []

    netg.train()
    netd.train()

    # for each epoch
    for epoch in range(9):
        for i, data in enumerate(dataloader):
            # data[0] = data[0].cuda()
            # data[2] = data[2].cuda()
            fake_high = netg(data[2])         
            # updata D
            optimizerD.zero_grad()
            fake_AB = torch.cat([data[2].float(), fake_high.float()], 1)  
            pred_fake = netd(fake_AB.detach())
            loss_D_fake = criterion(pred_fake, fake_lable)
            real_AB = torch.cat([data[2].float(), data[0].float()], 1)
            pred_real = netd(real_AB)
            loss_D_real = criterion(pred_real, real_lable)
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward(retain_graph=True)
            optimizerD.step()
            # updata G
            optimizerG.zero_grad()
            fake_AB = torch.cat([data[2].float(), fake_high.float()], 1) 
            pred_fake = netd(fake_AB)
            loss_G_GAN = criterion(pred_fake, real_lable)
            loss_G_L1 = torch.mean(torch.pow((fake_high - data[0]), 2))
            loss_G = 100*loss_G_L1 + loss_G_GAN
            loss_G.backward(retain_graph=True)
            optimizerG.step()
            G_LOSS.append(loss_G.item())
            D_LOSS.append(loss_D.item())
            # if i % 500 == 0 or loss_G <= 1:
            if i % 500 == 0:
                vutils.save_image(data[0], 'F:/CGAN/outf/{}_{}_high.png'.format(epoch + 1, i + 1), normalize= True)
                vutils.save_image(data[2], 'F:/CGAN/outf/{}_{}_low.png'.format(epoch + 1, i + 1), normalize= True)
                vutils.save_image(fake_high, 'F:/CGAN/outf/{}_{}.png'.format(epoch + 1, i + 1), normalize= True)
                if i % 5000 == 0:
                    torch.save({'epoch': epoch + 1, 'state_dict': netg.state_dict()}, 'F:/CGAN/outf/{}_{}_netG.pth'.format(epoch+1, i+1))
                    torch.save({'epoch': epoch + 1, 'state_dict': netd.state_dict()}, 'F:/CGAN/outf/{}_{}_netD.pth'.format(epoch+1, i+1))
            print('EPOCH[{}/10]IMG[{}/{}]\tloss_d:{:.4f},loss_g:{:.4f}'.format(epoch+1, i+1, len(dataloader), loss_D, loss_G))

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_LOSS,label="G")
    plt.plot(D_LOSS,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss.jpg')
    plt.show()

def test():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = NetG().to(device)
    model_dict = torch.load('9_30001_netG.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict['state_dict'])
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    dataset = ImageFolder('input', transform= transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=int(2),
                                                 drop_last=True)
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            out = model(data[0])
            vutils.save_image(out, 'F:/CGAN/outf/{}.png'.format(i+1), normalize= True)
            vutils.save_image(data[0], 'F:/CGAN/outf/{}_gt.png'.format(i+1), normalize= True)



if __name__ == '__main__':
    # train()
    test()