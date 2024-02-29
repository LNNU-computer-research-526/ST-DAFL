#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import argparse
import os
import numpy as np
import math
import sys
import pdb

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets.mnist import MNIST
from lenet import LeNet5Half
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import resnet
import random
from datafree.synthesis import BackdoorSuspectLoss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--teacher_dir', type=str, default='/cache/models/')
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.02, help='learning rate')
parser.add_argument('--lr_S', type=float, default=0.1, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=1000, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--oh', type=float, default=0.05, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.01, help='activation loss')
parser.add_argument('--output_dir', type=str, default='/cache/models/')
parser.add_argument('--shufl_coef', type=float, default=0.1)
parser.add_argument('--mb_rate', type=float, default=0.01)

opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True 

accr = 0
accr_best = 0

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(opt.channels, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img
        
generator = Generator().cuda()
    
teacher = torch.load(opt.teacher_dir + 'teacher').cuda()
teacher.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()

teacher = nn.DataParallel(teacher)
generator = nn.DataParallel(generator)

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl

if opt.dataset == 'MNIST':    
    # Configure data loader   
    net = LeNet5Half().cuda()
    net = nn.DataParallel(net)
    data_test = MNIST(opt.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))           
    data_test_loader = DataLoader(data_test, batch_size=64, num_workers=0, shuffle=False)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
    optimizer_S = torch.optim.Adam(net.parameters(), lr=opt.lr_S)

if opt.dataset != 'MNIST':  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if opt.dataset == 'cifar10': 
        net = resnet.ResNet18().cuda()
        net = nn.DataParallel(net)
        data_test = CIFAR10(opt.data,
                          train=False,
                          transform=transform_test)
    if opt.dataset == 'cifar100': 
        net = resnet.ResNet18(num_classes=100).cuda()
        net = nn.DataParallel(net)
        data_test = CIFAR100(opt.data,
                          train=False,
                          transform=transform_test)
    data_test_loader = DataLoader(data_test, batch_size=opt.batch_size, num_workers=0)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)

    optimizer_S = torch.optim.SGD(net.parameters(), lr=opt.lr_S, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch, learing_rate):
    if epoch < 800:
        lr = learing_rate
    elif epoch < 1600:
        lr = 0.1*learing_rate
    else:
        lr = 0.01*learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def cosine_similarity(m1, m2):
    global dot_product

    for i in range(m1.shape[0]):
        t = abs(np.linalg.norm(m1[i],axis=0))
        st = abs(np.linalg.norm(m2[i],axis=0))
        dot = np.dot(m1[i],m2[i])
        dot_product.append(dot/(t*st))
    return 0
def fill_signal(dp):
    global signal
    temp = sorted(dp,reverse=True)
    threshold = temp[100]
    for i in range(len(temp)):
        if dp[i] >= threshold:
            signal[i] = 1
    return 0
def save_imgs(x):
    global temp_x,flagx
    if flagx:
        temp_x = x
        flagx = False
    else:
        temp_x = np.vstack([temp_x, x])
def fill_mb():
    global memory_bank,signal,temp_x
    for i in range(len(signal)):
        if signal[i] == 1:
            memory_bank.append(temp_x[i])
    return 0
# ----------
#  Training
# ----------
flag_mb = False
suspect_loss = BackdoorSuspectLoss(teacher, coef=opt.shufl_coef)
memory_bank = []
for epoch in range(opt.n_epochs):
    total_correct = 0
    avg_loss = 0.0
    suspect_loss.prepare_select_shuffle()
    shuffle_teacher = suspect_loss.make_shuffle_suspect(teacher)
    if opt.dataset != 'MNIST':
        adjust_learning_rate(optimizer_S, epoch, opt.lr_S)

    for i in range(120):
        temp_x = []
        flagx = True
        dot_product = []
        net.train()
        z = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        optimizer_G.zero_grad()
        optimizer_S.zero_grad()        
        gen_imgs = generator(z)
        save_imgs(gen_imgs.detach().cpu().numpy())
        if flag_mb:
            great_imgs = torch.Tensor(random.sample(memory_bank, 127),requires_grad=True).cuda()
            outputs_T, features_T = teacher(great_imgs, out_feature=True)
            pred = outputs_T.data.max(1)[1]
            loss_activation = -features_T.abs().mean()
            loss_one_hot = criterion(outputs_T, pred)
            softmax_o_T = torch.nn.functional.softmax(outputs_T, dim=1).mean(dim=0)
            loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
            loss = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a
            loss_kd = kdloss(net(great_imgs.detach()), outputs_T.detach())
            loss += loss_kd
            loss.backward()
            optimizer_G.step()
            optimizer_S.step()


        outputs_T, features_T = teacher(gen_imgs, out_feature=True)
        outputs_ST = shuffle_teacher(gen_imgs)
        cosine_similarity(outputs_T.cpu().detach().numpy(), outputs_ST.cpu().detach().numpy())
        # signal = [0 for t in range(len(dot_product))]
        # fill_signal(dot_product)
        signal = [1 for t in range(len(dot_product))]


        memory_bank = []
        fill_mb()
        if not flag_mb:
            flag_mb = True
        if i == 1:
            print ("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, opt.n_epochs,loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))
    print('mb、dp、x',len(memory_bank),len(dot_product),len(temp_x))


    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images = images.cuda()
            labels = labels.cuda()
            net.eval()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test_loader)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
    accr = round(float(total_correct) / len(data_test), 4)
    if accr > accr_best:
        # torch.save(net,opt.output_dir + 'student')
        accr_best = accr
print('acc:',accr_best)
