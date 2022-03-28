

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from dcgan_model import Generator, Discriminator, Discriminator_SNGAN
#from alexnet import AlexNet
import tensorflow as tf
import torchvision
import pandas as pd

from PIL import Image
from models import *

writer = SummaryWriter()


def filter_indices(trainset):
    index_list = []
    print("indices = ", classes_indices)
    for i in range(len(trainset)):
        if trainset[i][1] in classes_indices:
           index_list.append(i)
    return index_list


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
                img[0] = img[1]
                img[2] = img[1]
                return img, dummy_label
            else:
                return img, dummy_label



if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--train_with_teacher', action='store_true', help='enables teacher')
    parser.add_argument('--dataset', type=str, default='svhn', help='Dataset used to train GAN')
    parser.add_argument('--network', type=str, default='resnet', help='Network used to train GAN')
    parser.add_argument('--sngan', action='store_true', help='enable sngan')
    parser.add_argument('--proxy_ds_name', type=str, default='40_class', help='cifar-100 40 classes')
    parser.add_argument('--teacher_path', type=str, default='', help='teacher model path')
    parser.add_argument('--num_classes', type=int, default=10, help='num classes in teacher model')
    parser.add_argument('--synthetic_dir', type=str, default='', help='Dataset Dir for training GAN')
    parser.add_argument('--total_synth_samples', default=50000, type=int, help='num of synthetic samples')
    parser.add_argument('--grey_scale', action='store_true', help="grey scale images to train student")

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if opt.dataset=='svhn':
        dataset =  torchvision.datasets.SVHN(root=opt.dataroot, split='train', download=True, transform=transform_train)
    elif opt.dataset=='cifar100':
        trainset =  torchvision.datasets.CIFAR100(
                    root=opt.dataroot, train=True, download=True, transform=transform_train)

        print(trainset.class_to_idx)
        id_to_class_mapping = {}
        for cl, idx in trainset.class_to_idx.items():
            id_to_class_mapping[idx] = cl
        print(id_to_class_mapping)

        if opt.proxy_ds_name == '40_class':
            classes_set = {'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
            'bottle', 'bowl', 'can', 'cup', 'plate',
            'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
            'clock', 'keyboard', 'lamp', 'telephone', 'television',
            'bed', 'chair', 'couch', 'table', 'wardrobe',
            'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
            'bridge', 'castle', 'house', 'road', 'skyscraper',
            'cloud', 'forest', 'mountain', 'plain', 'sea'}
        elif opt.proxy_ds_name == '6_class':
            classes_set = {'road', 'cloud', 'forest', 'mountain', 'plain', 'sea'}
        else:
            classes_set = {'plate', 'rose', 'castle', 'keyboard', 'house', 'forest', 'road', 'television', 'bottle', 'wardrobe'}
        classes_indices = []
        for k in classes_set:
            classes_indices.append(trainset.class_to_idx[k])
        print(classes_indices)

        index_list = filter_indices(trainset)

        dataset = torch.utils.data.Subset(trainset, index_list)
        print(len(dataset))

    elif opt.dataset=='cifar10':
        dataset =  torchvision.datasets.CIFAR10(
                    root=opt.dataroot, train=True, download=True, transform=transform_train)
        print("cifar-10", len(dataset))

    elif opt.dataset=='synthetic':
        arr = os.listdir(opt.synthetic_dir)
        data = pd.DataFrame(data=arr, columns=['filename'])
        csv_filename = 'synthetic_data.csv'
        data.to_csv(csv_filename)
        print(csv_filename)
        dir_name = opt.synthetic_dir
        dataset = SyntheticDataset(csv_file=csv_filename, root_dir=dir_name, transform=transform_train, length=int(opt.total_synth_samples), grey=opt.grey_scale)


    """
    dataset = dset.CIFAR100(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    """

    nc=3

    assert dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))


    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)
   
    """
    netC = AlexNet(ngpu).to(device)
    #netC.load_state_dict(torch.load('../../train_student_KD/AlexNet/CIFAR10_data/best_model.pth'))
    netC.load_state_dict(torch.load('CIFAR_10/network/best_model.pth'))
    
    print(netC)
    netC.eval()
    """
    if str(opt.network)=='resnet':
        netC = ResNet18(opt.num_classes)
    else:
        netC = AlexNet()
    netC = netC.to(device)
    netC = torch.nn.DataParallel(netC)
    state = {
            'net': netC.state_dict(),
            'acc': 90,
            'epoch': 200,
        }
    
    state = torch.load(opt.teacher_path)
    print(state['acc'])
    netC.load_state_dict(state['net'])
    netC.eval()


    if opt.sngan==False:
        netD = Discriminator(ngpu).to(device)
    else:
        netD = Discriminator_SNGAN(ngpu).to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()
    criterion_sum = nn.BCELoss(reduction = 'sum')

    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    threshold = []
    """ 
    # Used classes of CIFAR-100 (Background classes used here)
    inc_classes = [68, 23, 33, 49, 60, 71]
    """
    for epoch in range(opt.niter):
        num_greater_thresh = 0
        count_class = [0]*10
        count_class_less = [0]*10
        count_class_hist = [0]*10
        count_class_less_hist = [0]*10
        classification_loss_sum = 0
        errD_real_sum = 0
        errD_fake_sum = 0
        errD_sum = 0
        errG_adv_sum = 0
        data_size = 0 
        accD_real_sum = 0
        accD_fake_sum = 0
        accG_sum = 0
        accD_sum = 0
        div_loss_sum = 0
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            #real_cpu = torch.from_numpy(data[0].numpy()[np.isin(data[1],inc_classes)]).to(device)
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            if batch_size==0:
                continue
            data_size = data_size + batch_size
            label = torch.full((batch_size,), real_label, device=device)
            label = label.type(torch.cuda.FloatTensor)
            output = netD(real_cpu)
            
            errD_real = criterion(output, label)
            errD_real_sum = errD_real_sum + (criterion_sum(output,label)).cpu().data.numpy()

            accD_real = (label[output>0.5]).shape[0]
            accD_real_sum = accD_real_sum + float(accD_real)

            errD_real.backward()
            
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            #print(fake.shape, noise.shape)
            fake_class = netC(fake)
            sm_fake_class = F.softmax(fake_class, dim=1)
            
            class_max = fake_class.max(1,keepdim=True)[0]
            class_argmax = fake_class.max(1,keepdim=True)[1]
            
            if opt.train_with_teacher: 
                # Classification loss
                classification_loss = torch.mean(torch.sum(-sm_fake_class*torch.log(sm_fake_class+1e-5),dim=1))
                classification_loss_add = torch.sum(-sm_fake_class*torch.log(sm_fake_class+1e-5))
                classification_loss_sum = classification_loss_sum + (classification_loss_add).cpu().data.numpy() 
            
                sm_batch_mean = torch.mean(sm_fake_class,dim=0)
                div_loss = torch.sum(sm_batch_mean*torch.log(sm_batch_mean)) # Maximize entropy across batch
                div_loss_sum = div_loss_sum + div_loss.item()*batch_size
            
            label.fill_(fake_label)
            output = netD(fake.detach())

            errD_fake = criterion(output, label)
            errD_fake_sum = errD_fake_sum + (criterion_sum(output, label)).cpu().data.numpy()

            accD_fake = (label[output<=0.5]).shape[0]
            accD_fake_sum = accD_fake_sum + float(accD_fake)

            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            errD_sum = errD_real_sum + errD_fake_sum

            accD = accD_real + accD_fake
            accD_sum = accD_real_sum + accD_fake_sum

            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            if opt.train_with_teacher:
                c_l = 50
                d_l = 500
            else:
                c_l = 0 # Hyperparameter to weigh entropy loss
                d_l = 0 # Hyperparameter to weigh the diversity loss
            errG_adv = criterion(output, label) 
            errG_adv_sum = errG_adv_sum + (criterion_sum(output, label)).cpu().data.numpy()

            accG = (label[output>0.5]).shape[0]
            accG_sum = accG_sum + float(accG)

            #errG = errG_adv + c_l * classification_loss + d_l * div_loss
            errG = errG_adv 
            errG_sum = errG_adv_sum + c_l * classification_loss_sum + d_l * div_loss_sum
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            """
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            """
            pred_class = F.softmax(fake_class,dim=1).max(1, keepdim=True)[0]
            pred_class_argmax = F.softmax(fake_class,dim=1).max(1, keepdim=True)[1]
            num_greater_thresh = num_greater_thresh + (torch.sum(pred_class > 0.9).cpu().data.numpy())
            for argmax, val in zip(pred_class_argmax, pred_class):
                if val > 0.9:
                    count_class_hist.append(argmax)
                    count_class[argmax] = count_class[argmax] + 1
                else:
                    count_class_less_hist.append(argmax)
                    count_class_less[argmax] = count_class_less[argmax] + 1

            if i % 100 == 0:
                #tf.summary.image("Gen Imgs Training", (fake+1)/2, epoch)
                grid = torchvision.utils.make_grid((fake+1)/2)
                writer.add_image("Gen Imgs Training", grid, epoch)
        
        # do checkpointing
        if epoch>=198:
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))  

        # Generate fake samples for visualization

        test_size = 100
        noise_test = torch.randn(test_size, nz, 1, 1, device=device)
        fake_test = netG(noise_test)
        fake_test_class = netC(fake_test)
        pred_test_class_max = F.softmax(fake_test_class,dim=1).max(1, keepdim=True)[0]
        pred_test_class_argmax = F.softmax(fake_test_class,dim=1).max(1, keepdim=True)[1]

        """
        for i in range(10):
            print("Score>0.9: Class",i,":",torch.sum(((pred_test_class_argmax.view(test_size)==i) & (pred_test_class_max.view(test_size)>0.9)).float()))
            print("Score<0.9: Class",i,":",torch.sum(((pred_test_class_argmax.view(test_size)==i) & (pred_test_class_max.view(test_size)<0.9)).float()))
        """
 
        if fake_test[pred_test_class_argmax.view(test_size)==0].shape[0] > 0:
            grid = torchvision.utils.make_grid((fake_test[pred_test_class_argmax.view(test_size)==0]+1)/2)
            writer.add_image("Gen Imgs Test: Airplane", grid, epoch)

        if fake_test[pred_test_class_argmax.view(test_size)==1].shape[0] > 0:
            grid = torchvision.utils.make_grid((fake_test[pred_test_class_argmax.view(test_size)==1]+1)/2)
            writer.add_image("Gen Imgs Test: Automobile", grid, epoch)

        if fake_test[pred_test_class_argmax.view(test_size)==2].shape[0] > 0:
            grid = torchvision.utils.make_grid((fake_test[pred_test_class_argmax.view(test_size)==2]+1)/2)
            writer.add_image("Gen Imgs Test: Bird", grid, epoch)

        if fake_test[pred_test_class_argmax.view(test_size)==3].shape[0] > 0:
            grid = torchvision.utils.make_grid((fake_test[pred_test_class_argmax.view(test_size)==3]+1)/2)
            writer.add_image("Gen Imgs Test: Cat", grid, epoch)

        if fake_test[pred_test_class_argmax.view(test_size)==4].shape[0] > 0:
            grid = torchvision.utils.make_grid((fake_test[pred_test_class_argmax.view(test_size)==4]+1)/2)
            writer.add_image("Gen Imgs Test: Deer", grid, epoch)

        if fake_test[pred_test_class_argmax.view(test_size)==5].shape[0] > 0:
            grid = torchvision.utils.make_grid((fake_test[pred_test_class_argmax.view(test_size)==5]+1)/2)
            writer.add_image("Gen Imgs Test: Dog", grid, epoch)

        if fake_test[pred_test_class_argmax.view(test_size)==6].shape[0] > 0:
            grid = torchvision.utils.make_grid((fake_test[pred_test_class_argmax.view(test_size)==6]+1)/2)
            writer.add_image("Gen Imgs Test: Frog", grid, epoch)

        if fake_test[pred_test_class_argmax.view(test_size)==7].shape[0] > 0:
            grid = torchvision.utils.make_grid((fake_test[pred_test_class_argmax.view(test_size)==7]+1)/2)
            writer.add_image("Gen Imgs Test: Horse", grid, epoch)

        if fake_test[pred_test_class_argmax.view(test_size)==8].shape[0] > 0:
            grid = torchvision.utils.make_grid((fake_test[pred_test_class_argmax.view(test_size)==8]+1)/2)
            writer.add_image("Gen Imgs Test: Ship", grid, epoch)

        if fake_test[pred_test_class_argmax.view(test_size)==9].shape[0] > 0:
            grid = torchvision.utils.make_grid((fake_test[pred_test_class_argmax.view(test_size)==9]+1)/2)
            writer.add_image("Gen Imgs Test: Truck", grid, epoch)
        """
        print(count_class , "Above  0.9")
        print(count_class_less, "Below 0.9")
        """
        writer.add_histogram("above 0.9", np.asarray(count_class), epoch, bins=10)
        writer.add_histogram("above 0.9", np.asarray(count_class), epoch, bins=10)
        threshold.append(num_greater_thresh)
        
        writer.add_scalar("1 Train Discriminator accuracy(all)", accD_sum/ (2*data_size), epoch)
        writer.add_scalar("2 Train Discriminator accuracy(fake)", accD_fake_sum/ data_size, epoch)
        writer.add_scalar("3 Train Discriminator accuracy(real)", accD_real_sum/ data_size, epoch)
        writer.add_scalar("4 Train Generator accuracy(fake)", accG_sum/ data_size, epoch)
        writer.add_scalar("5 Train Discriminator loss (real)", errD_real_sum/ data_size, epoch)
        writer.add_scalar("6 Train Discriminator loss (fake)", errD_fake_sum/ data_size, epoch)
        writer.add_scalar("7 Train Discriminator loss (all)", errD_sum/(2* data_size), epoch)
        writer.add_scalar("8 Train Generator loss (adv)", errG_adv_sum/ data_size, epoch)
        writer.add_scalar("9 Train Generator loss (classification)", classification_loss_sum/ data_size, epoch)
        writer.add_scalar("10 Train Generator loss (diversity)", div_loss_sum/ data_size, epoch)
        writer.add_scalar("11 Train Generator loss (all)", errG_sum/ data_size, epoch)

        writer.export_scalars_to_json("./all_scalars.json")
   
        """  
        if epoch%50==0:
            for img_num in range(50000):
                test_size = 1
                noise_test = torch.randn(test_size, nz, 1, 1, device=device)
                img = netG(noise_test)
                #print(img.shape)

                img = img[0].detach().cpu()
                img = img / 2 + 0.5   # unnormalize
                npimg = img.numpy()   # convert from tensor
                np_img = np.transpose(npimg, (1, 2, 0))

                np_img = np_img*255
                np_img = np_img.astype(np.uint8)
                #print(x, np.max(x), np.min(x))
                im = Image.fromarray(np_img)
                im.save("./SVHN/svhn_generated_images/file_" + str(img_num)+ ".png")
        """

writer.close()

