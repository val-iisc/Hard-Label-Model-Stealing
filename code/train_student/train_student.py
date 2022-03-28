
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

import random
from models import *
from utils import progress_bar

import pytorch_warmup as warmup
import pickle


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def filter_indices(trainset, classes_indices):
    index_list = []
    print("indices = ", classes_indices)
    for i in range(len(trainset)):
        if trainset[i][1] in classes_indices:
           index_list.append(i)
    return index_list


from torch.autograd import Variable
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dcgan_netG', default='', help="path to dcgan netG ")
parser.add_argument('--div_gan_netG', default='', help="path to diverse gan netG ")
parser.add_argument('--network', default='resnet', type=str, help="resnet or alexnet")
parser.add_argument('--student_path', default='', help="path to student model")

parser.add_argument('--dcgan_netG_pickle', action='store_true', help="use diverse samples from pickle of dcgan")
parser.add_argument('--div_gan_netG_pickle', action='store_true', help="use diverse samples from pickle of diverse gan")
parser.add_argument('--dcgan_pickle_path', default='', type=str, help="path to pickle data")
parser.add_argument('--dcgan_labels_pickle_path', default='', type=str, help="path to pickle data labels")
parser.add_argument('--div_gan_pickle_path', default='', type=str, help="path to pickle data")

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--count', default=1, type=int, help='count student training ')
parser.add_argument('--max_epochs', default=200, type=int, help='max epochs')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

parser.add_argument('--pad_crop', action='store_true', help="pad crop")
parser.add_argument('--mixup', action='store_true', help="mixup")
parser.add_argument('--proxy_data', action='store_true', help="use proxy data ")
parser.add_argument('--proxy_data_ratio', default=0, type=float, help="use proxy data ratio")
parser.add_argument('--dcgan_data', action='store_true', help="use dcgan data")
parser.add_argument('--dcgan_data_ratio', default=0, type=float, help="use dcgan data ratio")
parser.add_argument('--div_gan_data', action='store_true', help="use div_gan data")
parser.add_argument('--div_gan_data_ratio', default=0, type=float, help="use div_gan data ratio")

parser.add_argument('--from_scratch', action='store_true', help="student trained from scratch")
parser.add_argument('--cutmix', action='store_true', help="use cutmix for training")
parser.add_argument('--beta', default=1, type=float, help='beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float, help='cutmix_prob')
parser.add_argument('--auto-augment', default=False, type=bool, help='auto-augment')
parser.add_argument('--name', default='', type=str, help='name of student model')
parser.add_argument('--warmup', action='store_true', help='use warmup ')

parser.add_argument('--train_with_dcgan', action='store_true', help='train student model with gan images ')
parser.add_argument('--train_with_div_gan', action='store_true', help='train student model with gan images ')
parser.add_argument('--proxy_ds_name', default='40_class', type=str, help='cifar-100 proxy dataset')

parser.add_argument('--val_data_dcgan', default='', type=str, help='name of dcgan val data')
parser.add_argument('--teacher_path', default='', type=str, help='name of teacher model')

parser.add_argument('--proxy_dataset', default='cifar100', type=str, help='proxy dataset')
parser.add_argument('--true_dataset', default='cifar10', type=str, help='true dataset')
parser.add_argument('--num_classes', default=10, type=int, help='num of classes of teacher model')
parser.add_argument('--total_synth_samples', default=50000, type=int, help='num of synthetic samples')
parser.add_argument('--synthetic_dir', default='', type=str, help='path of synthetic dir')
parser.add_argument('--grey_scale', action='store_true', help="grey scale images to train student")
parser.add_argument('--wo_batch_norm', action='store_true', help="train student removing batch norm layers")


args = parser.parse_args()
print(args)

def get_model_accuracy(net, test_loader_):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader_):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    return acc


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

class SyntheticDataset(Dataset):
    """Synthetic dataset."""

    def __init__(self, csv_file, root_dir, transform=None, length=50000, grey=False):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.length = length
        self.grey = grey

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dummy_label = 0
        img_name = os.path.join(self.root_dir, self.csv_file.iloc[idx]['filename'])
        img = Image.open(img_name)
       
        if self.transform:
            img = self.transform(img)

        if self.grey==True:
            return img
        else:
            return img, dummy_label

if args.val_data_dcgan!='':
    with open(args.val_data_dcgan,'rb') as f:
        val_data_dcgan = pickle.load(f)
    val_loader = torch.utils.data.DataLoader(
                    val_data_dcgan, batch_size=128, shuffle=True, num_workers=2)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_val_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

max_epochs = int(args.max_epochs)
batch_size = 128


ngpu=1
#gen_path = "./out_svhn_step2/netG_epoch_99.pth"
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



max_samples = 50000
nz=100
test_size = 10

##  Load data in data array ##

data = torch.zeros((max_samples,3,32,32))
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


if args.proxy_data==True:

    if args.proxy_dataset == 'cifar100':
        trainset =  torchvision.datasets.CIFAR100(
                    root='./data/', train=True, download=True, transform=transform_train)

        print(trainset.class_to_idx)
        print(len(trainset.targets), len(trainset.data), len(trainset))
        print(trainset.data.shape, type(trainset.data))

        id_to_class_mapping = {}
        for cl, idx in trainset.class_to_idx.items():
            id_to_class_mapping[idx] = cl
        print(id_to_class_mapping)
        
        if args.proxy_ds_name == '40_class':
            classes_set = {'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
            'bottle', 'bowl', 'can', 'cup', 'plate',
            'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
            'clock', 'keyboard', 'lamp', 'telephone', 'television',
            'bed', 'chair', 'couch', 'table', 'wardrobe',
            'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
            'bridge', 'castle', 'house', 'road', 'skyscraper',
            'cloud', 'forest', 'mountain', 'plain', 'sea'}
        elif args.proxy_ds_name == '6_class':
            classes_set = {'road', 'cloud', 'forest', 'mountain', 'plain', 'sea'}
        else:
            classes_set = {'plate', 'rose', 'castle', 'keyboard', 'house', 'forest', 'road', 'television', 'bottle', 'wardrobe'}

        classes_indices = []
        for k in classes_set:
            classes_indices.append(trainset.class_to_idx[k])
        print(classes_indices)

        index_list = filter_indices(trainset, classes_indices)
        print(len(index_list))
        trainset_1 = torch.utils.data.Subset(trainset, index_list)
    
    elif args.proxy_dataset == 'cifar10':
        trainset =  torchvision.datasets.CIFAR10(
                root='./data/', train=True, download=True, transform=transform_train)
        trainset_1 = trainset

    elif args.proxy_dataset == 'synthetic':
        arr = os.listdir(args.synthetic_dir)
        #random.shuffle(arr)
        synthetic_data = pd.DataFrame(data=arr, columns=['filename'])
        csv_filename = 'synthetic_data.csv'
        synthetic_data.to_csv(csv_filename)
        print(csv_filename)

        dir_name = args.synthetic_dir
        trainset = SyntheticDataset(csv_file=csv_filename, root_dir=dir_name, transform=transform_train, length=int(args.total_synth_samples), grey=args.grey_scale)
        trainset_1 = trainset

    num_train = int(len(trainset_1))
    proxy_samples = min(num_train, int(float(args.proxy_data_ratio)*max_samples))
    print("proxy samples = ", proxy_samples)

    if args.proxy_dataset == 'cifar100':
        for idx in range(0, proxy_samples):
            if trainset_1[idx][1] not in classes_indices:
                print("False")
    for idx in range(0, proxy_samples):
        data[idx] = trainset_1[idx][0]
        #print(data[idx])
    print("fine")
    
else:
    proxy_samples = 0

if args.dcgan_data==True:
    dcgan_samples = int(float(args.dcgan_data_ratio)*max_samples)
    if args.dcgan_netG_pickle==True:
        with open(args.dcgan_pickle_path,'rb') as f:
            x = pickle.load(f)
        data[proxy_samples: (proxy_samples+dcgan_samples)] = x[0:dcgan_samples]
    else:
        for idx in range(int(dcgan_samples/test_size)):
            noise_test = torch.randn(test_size, nz, 1, 1, device=device)
            imgs = dcgan_netG(noise_test)
            data[(proxy_samples + idx*test_size) : (proxy_samples + idx*test_size + test_size)] = imgs.detach().cpu()
else:
    dcgan_samples = 0

if args.div_gan_data==True:
    div_gan_samples = int(float(args.div_gan_data_ratio)*max_samples)
    if args.div_gan_netG_pickle==True:
        with open(args.div_gan_pickle_path,'rb') as f:
            x = pickle.load(f)
        data[(proxy_samples + dcgan_samples): (proxy_samples + dcgan_samples + div_gan_samples)] = x[0:div_gan_samples]
    else:
        for idx in range(int(div_gan_samples/test_size)):
            noise_test = torch.randn(test_size, nz, 1, 1, device=device)
            imgs = div_gan_netG(noise_test)
            data[(proxy_samples  + dcgan_samples + idx*test_size) : (proxy_samples + dcgan_samples + idx*test_size + test_size)] = imgs.detach().cpu()
else:
    div_gan_samples = 0



##  Load the student and teacher models  ##

if args.network=='resnet':
    teacher_net = ResNet18(args.num_classes)
else:
    teacher_net = AlexNet()

teacher_net = teacher_net.to(device)
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

if args.network=='resnet':
    student_net = ResNet18(args.num_classes)
else:
    if args.wo_batch_norm==True:
        student_net = AlexNet_half_wo_BN()
    else:
        student_net = AlexNet_half()
student_net = student_net.to(device)
state = {
           'net': student_net.state_dict(),
           'acc': 90,
           'epoch': 200,
        }

if args.from_scratch==False:
    state = torch.load(str(args.student_path))
    print("Student val Acc : ", state['acc'])
    student_net.load_state_dict(state['net'])

teacher_net.eval()


##  Save the predictions from the teacher model in targets ##

target_arr = torch.zeros((len(data)))

if args.dcgan_labels_pickle_path=='':
    for i in range(len(data)):
        inputs = torch.reshape(data[i], (1,3,32,32))
        teacher_outputs = teacher_net(inputs)
        teacher_outputs = teacher_outputs.detach()
        _, teacher_predicted = teacher_outputs.max(1)
        target_arr[i] = teacher_predicted.item()
else:
    with open(args.dcgan_labels_pickle_path,'rb') as f:
        x = pickle.load(f)
    target_arr[0: dcgan_samples] = x[0:dcgan_samples]

transform_train_pad_crop = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
if args.auto_augment:
    pil_transform = transforms.Compose([
        transforms.ToPILImage(),
        AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])

class GeneratedDataset_(Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if args.pad_crop==True:
            img = transform_train_pad_crop(self.data[idx])
        else:
            img = self.data[idx]
        if args.auto_augment==True:
            img = (img * 0.5 + 0.5)
            img = pil_transform(img)
        return img, self.targets[idx]


print('==> Preparing data..')


if args.proxy_data==True:
    trainset1 = GeneratedDataset_(data = data[0:proxy_samples], targets = target_arr[0:proxy_samples], transform=transform_train)

    loader_batch = int(batch_size*float(args.proxy_data_ratio))
    trainloader1 = torch.utils.data.DataLoader(
        trainset1, batch_size=loader_batch, shuffle=True, num_workers=2)

    iters_proxy = int(len(trainset1)/loader_batch) + 1
    print("training samples in proxy data: ", len(trainset1))


if args.dcgan_data==True:
    trainset2 = GeneratedDataset_(data = data[proxy_samples:proxy_samples+dcgan_samples], 
        targets = target_arr[proxy_samples:proxy_samples+dcgan_samples], transform=transform_train)

    loader_batch = int(batch_size*float(args.dcgan_data_ratio))
    trainloader2 = torch.utils.data.DataLoader(
        trainset2, batch_size=loader_batch, shuffle=True, num_workers=2)

    iters_dcgan = int(len(trainset2)/loader_batch) + 1
    print("training samples in dcgan data : ", len(trainset2))


if args.div_gan_data==True:
    trainset3 = GeneratedDataset_(data = data[proxy_samples+dcgan_samples:proxy_samples+dcgan_samples+div_gan_samples],
        targets = target_arr[proxy_samples+dcgan_samples:proxy_samples+dcgan_samples+div_gan_samples], transform=transform_train)

    loader_batch = int(batch_size*float(args.div_gan_data_ratio))
    trainloader3 = torch.utils.data.DataLoader(
        trainset3, batch_size=loader_batch, shuffle=True, num_workers=2)

    iters_div_gan = int(len(trainset3)/loader_batch) + 1
    print("training samples in div_gan data : ", len(trainset3))



if args.true_dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(
        root='./data/', train=False, download=True, transform=transform_test)
elif args.true_dataset == 'cifar100':
    testset = torchvision.datasets.CIFAR100(
        root='./data/', train=False, download=True, transform=transform_test)
    
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

if args.from_scratch==False:
    print("Student Start Acc = ", get_model_accuracy(student_net, testloader))

print("Calculated Teacher Acc = ", get_model_accuracy(teacher_net, testloader))


#max_iterations = 50000
#iters_in_one_epoch = int(num_train/batch_size)+1
#max_epochs = int(max_iterations/iters_in_one_epoch) + 1

print("max epochs = ", max_epochs)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(student_net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max_epochs)
iterations = 0
inital_lr = 0.01

fixed_tau_list = [0,0.99,0.999,0.9995,0.9998]
fixed_exp_avgs = []
fixed_tau_best_accs = [0,0,0,0,0,0,0]
best_tau_overall_acc = 0

for i in fixed_tau_list:
    fixed_exp_avgs.append(student_net.state_dict())

train_accs = []
val_accs = []
test_accs = []

def train(epoch, args):
    global iterations
    print('\nEpoch: %d' % epoch)
    teacher_net.eval()
    student_net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_batches = 1000000
    if args.proxy_data==True:
        it1 = iter(trainloader1)
        total_batches = min(total_batches, iters_proxy)
    if args.dcgan_data==True:
        it2 = iter(trainloader2)
        total_batches = min(total_batches, iters_dcgan)
    if args.div_gan_data==True:
        it3 = iter(trainloader3)
        total_batches = min(total_batches, iters_div_gan)

    #total_batches = int((proxy_samples+dcgan_samples+div_gan_samples)/batch_size) + 1
    print("total batches : ", total_batches)

    for batch_idx in range(total_batches):
        inputs1, targets1 = None, None
        inputs2, targets2 = None, None
        if args.proxy_data==True:
            inputs1, targets1 = it1.next()
            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            inputs = inputs1
            targets = targets1
        if args.dcgan_data==True:
            inputs2, targets2 = it2.next()
            inputs2, targets2 = inputs2.to(device), targets2.to(device)
            if args.proxy_data==True:
                inputs = torch.vstack((inputs, inputs2)) 
                targets = torch.cat((targets, targets2))
            else:
                inputs = inputs2
                targets = targets2
        if args.div_gan_data==True:
            inputs3, targets3 = it3.next()
            inputs3, targets3 = inputs3.to(device), targets3.to(device)
            if args.proxy_data==True or args.dcgan_data==True:
                inputs = torch.vstack((inputs, inputs3))
                targets = torch.cat((targets, targets3))
            else:
                inputs = inputs3
                targets = targets3
                

        targets = targets.type(torch.cuda.LongTensor)
        #print(inputs.shape, targets.shape)
        optimizer.zero_grad()
        teacher_outputs = teacher_net(inputs)
        teacher_outputs = teacher_outputs.detach()
        _, teacher_predicted = teacher_outputs.max(1)

        r = np.random.rand(1)
        if args.cutmix==True and args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets #teacher_predicted
            target_b = targets[rand_index] #teacher_predicted[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            student_outputs = student_net(inputs)
            loss = criterion(student_outputs, target_a) * lam + criterion(student_outputs, target_b) * (1. - lam) 

        if args.mixup==True:
            student_outputs = student_net(inputs)
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            loss = mixup_criterion(criterion, student_outputs, targets_a, targets_b, lam)
            _, student_predicted = torch.max(student_outputs.data, 1)
            correct += (lam * student_predicted.eq(targets_a.data).cpu().sum().float()
                             + (1 - lam) * student_predicted.eq(targets_b.data).cpu().sum().float())
            total += targets.size(0)
            #print(correct.item(), total, loss.item())
            loss.backward()
            optimizer.step()

        else:
            # compute output
            student_outputs = student_net(inputs)
            loss = criterion(student_outputs, targets)

        #print("targets, teacher's prediction : ", batch_idx, targets, teacher_predicted)
        if args.mixup==False:
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, student_predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += student_predicted.eq(targets).sum().item()
            #print("student prediction : ", student_predicted)

        iterations += 1
        for tau, new_state_dict in zip(fixed_tau_list, fixed_exp_avgs):
                for key, value in student_net.state_dict().items():
                    new_state_dict[key] = (1-tau)*value + tau*new_state_dict[key]


    if args.mixup==False:
        train_accs.append(100.*correct/total)
        print("Train Acc : ", 100.*correct/total)
    else:
        train_accs.append(100.*correct.item()/total)
        print("Train Acc : ", 100.*correct.item()/total)
    
    print("Learning rate : ", scheduler.get_lr())



def train_gan(epoch, args):
 
    transform_train_pad_crop = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
    if args.auto_augment:
        pil_transform = transforms.Compose([
            transforms.ToPILImage(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])

    total_batches = int(len(trainset_1)/batch_size) + 1
    global iterations
    print('\nEpoch: %d' % epoch)
    teacher_net.eval()
    student_net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_queries = 0
    for batch_idx in range(total_batches):
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        total_queries+=batch_size
        if args.train_with_dcgan==True:
            dcgan_netG.eval()
            gen_images = dcgan_netG(noise)
        else:
            div_gan_netG.eval()
            gen_images = div_gan_netG(noise)
        
        if args.auto_augment==True:
            imgs = (gen_images * 0.5 + 0.5)
            for im in range(len(imgs)):
                imgs[im] = transform_train_pad_crop(imgs[im])
                imgs[im] = pil_transform(imgs[im])
            else:
                imgs = gen_images
        inputs =  imgs.to(device)
        #print(inputs.shape, targets.shape)
        optimizer.zero_grad()
        teacher_outputs = teacher_net(inputs)
        teacher_outputs = teacher_outputs.detach()
        _, teacher_predicted = teacher_outputs.max(1)

        # compute output
        student_outputs = student_net(inputs)
        loss = criterion(student_outputs, teacher_predicted)
        #print("targets, teacher's prediction : ", batch_idx, targets, teacher_predicted)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, student_predicted = student_outputs.max(1)
        total += teacher_predicted.size(0)
        correct += student_predicted.eq(teacher_predicted).sum().item()
        #print("student prediction : ", student_predicted)

        iterations += 1
        for tau, new_state_dict in zip(fixed_tau_list, fixed_exp_avgs):
                for key, value in student_net.state_dict().items():
                    new_state_dict[key] = (1-tau)*value + tau*new_state_dict[key]

    print("Train Acc : ", 100.*correct/total)

    print(total_queries)
    print("Learning rate : ", scheduler.get_lr())


def test(epoch, args):
    global best_acc
    teacher_net.eval()
    student_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("Test Acc")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student_net(inputs)
            outputs_teacher = teacher_net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, predicted_teacher = outputs_teacher.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        print("Test Acc : ", 100.*correct/total)


    acc = 100.*correct/total
    test_accs.append(acc)
    if acc > best_acc:
        best_acc = acc
        state = {
            'net': student_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        """
        if args.dcgan_data==True:
            directory = './SVHN/train_student/checkpoint/' + str(args.network) + '/'+ str(args.count)
        else: 
            directory = './SVHN/train_student/checkpoint/' + str(args.network) + '/' +'div_gan/'+ str(args.count)
        if args.cutmix==True:
            directory = directory + "_cutmix"
        if not os.path.exists(directory):
            os.makedirs(directory) 
        print(directory + "/best_student_model_" + str(args.name) +".pth")
        torch.save(state, directory + "/best_student_model_" + str(args.name) +".pth")
        """
    print("Best Test Acc : ", best_acc)


def val(epoch, args):
    global best_val_acc
    global best_tau_overall_acc
    teacher_net.eval()
    student_net.eval()
    best_val_flag = False

    nb_classes = args.num_classes
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student_net(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    conf_num = confusion_matrix.diag()/confusion_matrix.sum(1)
    #print("confusion matrix : ", conf_num)
    conf_mat = conf_num.cpu().detach().numpy()
    conf_mat = conf_mat[~np.isnan(conf_mat)]
    acc = np.mean(conf_mat)*100
    print("Val acc : ", acc)
    val_accs.append(acc)
 
    if args.dcgan_data==True:
        directory = './train_student/checkpoint/' + str(args.network) + '/' + str(args.true_dataset) + '_' + str(args.proxy_dataset) + '/' + str(args.count)
    else:
        directory = './train_student/checkpoint/' + str(args.network) + '/' + str(args.true_dataset) + '_' + str(args.proxy_dataset) + '_div_gan/'+ str(args.count)
    if args.cutmix==True:
        directory = directory + "_cutmix"
    if not os.path.exists(directory):
        os.makedirs(directory)

    state = {
            'net': student_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        } 
    if acc > best_val_acc:
        best_val_acc = acc
        best_val_flag = True
        print(directory + "/best_val_student_model_" + str(args.name) +".pth")
        torch.save(state, directory + "/best_val_student_model_" + str(args.name) +".pth")
        print("Best Val Acc reached : ", best_val_acc)

    print(directory + "/last_epoch_best_student_model_" + str(args.name) +".pth")
    torch.save(state, directory + "/last_epoch_best_student_model_" + str(args.name) +".pth")


    if best_val_flag == True or epoch>=max_epochs-1:
        idx_count = 0
        for tau, new_state_dict in zip(fixed_tau_list, fixed_exp_avgs):
            nb_classes = args.num_classes
            confusion_matrix = torch.zeros(nb_classes, nb_classes)
            if args.network == 'resnet':
                evaluation_netC = ResNet18(args.num_classes)
            else:
                if args.wo_batch_norm==True:
                    evaluation_netC = AlexNet_half_wo_BN()
                else:
                    evaluation_netC = AlexNet_half()
            evaluation_netC.load_state_dict(new_state_dict)
            evaluation_netC = evaluation_netC.to(device)
            evaluation_netC.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = evaluation_netC(inputs)
                    _, preds = torch.max(outputs, 1)
                    for t, p in zip(targets.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

            conf_num = confusion_matrix.diag()/confusion_matrix.sum(1)
            conf_mat = conf_num.cpu().detach().numpy()
            conf_mat = conf_mat[~np.isnan(conf_mat)]
            div_gan_acc_tau = np.mean(conf_mat)*100
            if fixed_tau_best_accs[idx_count] < div_gan_acc_tau:
                fixed_tau_best_accs[idx_count] = div_gan_acc_tau
            idx_count += 1

        best_index = 0
        best_tau_div_gan_acc = 0
        for t in range(len(fixed_tau_list)):
            if best_tau_div_gan_acc < fixed_tau_best_accs[t]:
                best_tau_div_gan_acc = fixed_tau_best_accs[t]
                best_index = t

        if best_tau_overall_acc < fixed_tau_best_accs[best_index]:
            best_tau_overall_acc = fixed_tau_best_accs[best_index]

            if args.network == 'resnet':
                best_model_tau = ResNet18(args.num_classes)
            else:
                if args.wo_batch_norm==True:
                    best_model_tau = AlexNet_half_wo_BN()
                else:
                    best_model_tau = AlexNet_half()
            best_model_tau.load_state_dict(fixed_exp_avgs[best_index])
            best_model_tau = best_model_tau.to(device)
            best_model_tau.eval()
            total_test = 0
            correct_test = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = best_model_tau(inputs)
                    _, predicted = outputs.max(1)
                    total_test += targets.size(0)
                    correct_test += predicted.eq(targets).sum().item()

            print("Test Acc on ", args.true_dataset, " for best model of tau: ", best_index, best_tau_div_gan_acc, 100.*correct_test/total_test)
            state_1 = {
                        'net': best_model_tau.state_dict(),
                        'acc': 100.*correct_test/total_test,
                        'epoch': epoch,
                    }
            print(directory + "/best_tau_model_"+str(args.name)+'.pth')
            torch.save(state_1, directory + "/best_tau_model_"+str(args.name)+'.pth')
        


epoch=0
while epoch<max_epochs:
    if args.train_with_dcgan==True or args.train_with_div_gan==True:
        train_gan(epoch, args)
    else:
        train(epoch, args)
    val(epoch, args)
    test(epoch, args)
    print("Iterations = ", iterations)
    #if iterations>=max_iterations:
    #    break
    scheduler.step()
    if args.warmup==True and epoch<10:
        for param_group in optimizer.param_groups:
            param_group["lr"] = inital_lr * epoch
    epoch+=1

print("Best Test Acc = ", best_acc)
print("Best Val Acc : ", best_val_acc)

"""
with open('train_accs' + str(args.name) + '.pkl','wb') as f:
    pickle.dump(train_accs, f)

with open('val_accs' + str(args.name) + ' .pkl','wb') as f:
    pickle.dump(val_accs, f)

with open('test_accs' + str(args.name) + ' .pkl','wb') as f:
    pickle.dump(test_accs, f)

"""

