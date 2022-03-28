import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from dcgan_model import Generator, Discriminator
from auto_augment import AutoAugment

import pickle
import sys
sys.path.append('./')

from models import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--dcgan_netG', default='', help="path to dcgan netG ")
parser.add_argument('--div_gan_netG', default='', help="path to div_gan netG ")

parser.add_argument('--dcgan_out', default='', help="path to dcgan output ")
parser.add_argument('--div_gan_out', default='', help="path to dcgan output ")

parser.add_argument('--network', default='resnet', type=str, help="resnet or alexnet")
parser.add_argument('--teacher_path', default='', help="path to teacher model")
parser.add_argument('--num_classes', default=10, help="num classes for teacher model")


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

batch_size = 128

ngpu=1
if args.dcgan_netG!='':
    gen_path = str(args.dcgan_netG)
    dcgan_netG = Generator(ngpu).to(device)
    dcgan_netG.load_state_dict(torch.load(gen_path))
    dcgan_netG.eval()
    print(dcgan_netG)

if args.div_gan_netG!='':
    gen_path = str(args.div_gan_netG)
    div_gan_netG = Generator(ngpu).to(device)
    div_gan_netG.load_state_dict(torch.load(gen_path))
    print(div_gan_netG)
    div_gan_netG.eval()


if args.network=='resnet':
    teacher_net = ResNet18(int(args.num_classes))
else:
    teacher_net = AlexNet()

teacher_net = teacher_net.to(device)
#teacher_net.load_state_dict(torch.load('./checkpoint/ckpt.pth'))
if device == 'cuda':
    teacher_net = torch.nn.DataParallel(teacher_net)
    cudnn.benchmark = True
state = {
        'net': teacher_net.state_dict(),
        'acc': 90,
        'epoch': 200,
     }
state = torch.load(args.teacher_path)
print("Teacher Acc : ", state['acc'])
teacher_net.load_state_dict(state['net'])
teacher_net.eval()


class GeneratedDataset(Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.data[idx]
        return img, self.targets[idx]



## CREATE DCGAN VAL DATASET ##

if args.dcgan_netG!='':
    max_samples = 20000
    test_size = 10
    nz = 100
    data = torch.zeros((max_samples,3,32,32))
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    for idx in range(int(max_samples/test_size)):
        noise_test = torch.randn(test_size, nz, 1, 1, device=device)
        imgs = dcgan_netG(noise_test)
        data[(idx*test_size) : (idx*test_size + test_size)] = imgs.detach().cpu()

    target_arr = torch.zeros((len(data)))

    for i in range(len(data)):
        inputs = torch.reshape(data[i], (1,3,32,32))
        teacher_outputs = teacher_net(inputs)
        _, teacher_predicted = teacher_outputs.max(1)
        target_arr[i] = teacher_predicted.item()

    dcgan_dataset = GeneratedDataset(data = data[0:max_samples], targets = target_arr[0:max_samples], transform=transform_train)
    #with open('dcgan_val_data_cifar40.pkl','wb') as f:
    with open(args.dcgan_out,'wb') as f:
        pickle.dump(dcgan_dataset, f)

    with open(args.dcgan_out,'rb') as f:
        val_data_dcgan = pickle.load(f)
        val_loader = torch.utils.data.DataLoader(
                    val_data_dcgan, batch_size=128, shuffle=True, num_workers=2)


if args.div_gan_netG!='':
    
    max_samples = 20000
    test_size = 10
    nz = 100
    data = torch.zeros((max_samples,3,32,32))
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    for idx in range(int(max_samples/test_size)):
        noise_test = torch.randn(test_size, nz, 1, 1, device=device)
        imgs = div_gan_netG(noise_test)
        data[(idx*test_size) : (idx*test_size + test_size)] = imgs.detach().cpu()

    target_arr = torch.zeros((len(data)))

    for i in range(len(data)):
        inputs = torch.reshape(data[i], (1,3,32,32))
        teacher_outputs = teacher_net(inputs)
        _, teacher_predicted = teacher_outputs.max(1)
        target_arr[i] = teacher_predicted.item()

    div_gan_dataset = GeneratedDataset(data = data[0:max_samples], targets = target_arr[0:max_samples], transform=transform_train)
    with open(args.div_gan_out,'wb') as f:
        pickle.dump(div_gan_dataset, f)
