import os
import math
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST, SVHN, CelebA
import torchvision.transforms as T
import torchvision.models as models
from torchvision import models
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import argparse
import pandas as pd
import cv2
import csv

class CelebADataset(Dataset):

    def __init__(self, csv_file, root_dir, eval_file, targets, split='train', transform=None):

        self.attr = pd.read_csv(csv_file)
        self.eval_partition = pd.read_csv(eval_file)
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.targets = targets
        
        self.attr = self.attr[self.targets]
        self.attr = self.attr.replace(-1, 0)
        self.attr = self.attr.set_index('image_id')
        self.eval_partition = self.eval_partition.set_index('image_id')
        self.attr = self.attr.join(self.eval_partition)
        self.attr['image_id'] = self.attr.index
        
        if self.split == 'train':
            self.attr = self.attr.loc[self.attr['partition']==0]
            self.attr = self.attr.drop('partition', axis=1)
        else:
            self.attr = self.attr.loc[self.attr['partition']==2]
            self.attr = self.attr.drop('partition', axis=1)
            
        self.attr = self.attr[self.targets] 

    def __len__(self):
        return len(self.attr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        img_name = self.root_dir + self.attr.iloc[idx, 0]
        
        image = cv2.imread(img_name) / 255.0
        attr = self.attr.iloc[idx, 1:]
        attr = np.array([attr])
        attr = attr.astype('float')
        
        image = cv2.resize(image, dsize=(96, 96), interpolation=cv2.INTER_AREA)
        image = image.reshape(image.shape[2], image.shape[0], -1)
        sample = (torch.tensor(image).float(), attr)

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)
    
class MyDataset(Dataset):
    def __init__(self, dataset_name, train_flag, targets = None, transf=None):
        self.dataset_name = dataset_name
        
        if self.dataset_name == "cifar10":
            self.cifar10 = CIFAR10('../cifar10', train=train_flag, 
                                    download=True, transform=transf)
            
        if self.dataset_name == "celeba":
            attr_file = '../input/celeba-dataset/list_attr_celeba.csv'
            eval_file = '../input/celeba-dataset/list_eval_partition.csv'
            path = '../input/celeba-dataset/img_align_celeba/ali/'            
            self.celeba = CelebADataset(attr_file, path, eval_file, targets, split='train')  

    def __getitem__(self, index):
        if self.dataset_name == "cifar10":
            data, target = self.cifar10[index]
            
        if self.dataset_name == "celeba":
            data, target = self.celeba[index]
        
        return data, target, index

    def __len__(self):
        if self.dataset_name == "cifar10":
            return len(self.cifar10)
        
        if self.dataset_name == "celeba":
            return len(self.celeba)
        

def load_dataset(args, targets = None):
    
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) 
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) 
    ])
        
    if args['dataset'] == 'celeba':
        
        attr_file = '../input/celeba-dataset/list_attr_celeba.csv'
        eval_file = '../input/celeba-dataset/list_eval_partition.csv'
        path = '../input/celeba-dataset/img_align_celeba/img_align_celeba/'
        
        data_train = CelebADataset(attr_file, path, eval_file, targets, split='train')    
        data_unlabeled = MyDataset(args['dataset'], True, targets)
        data_test  = CelebADataset(attr_file, path, eval_file, targets, split='test')
        NO_CLASSES = len(targets) - 1
        adden = args['ADDENDUM']
        NUM_TRAIN = len(data_train)        
        no_train = NUM_TRAIN
        
    return data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train

class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, expansion, stride=1):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, expansion=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 1, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 1, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 1, stride=2)
        
        self.linear1 = nn.Linear(512*expansion, 2)
        self.linear2 = nn.Linear(512*expansion, 2)
        self.linear3 = nn.Linear(512*expansion, 2)
        self.linear4 = nn.Linear(512*expansion, 2)
        self.linear5 = nn.Linear(512*expansion, 2)
        self.linear6 = nn.Linear(512*expansion, 2)
        self.linear7 = nn.Linear(512*expansion, 2)
        self.linear8 = nn.Linear(512*expansion, 2)
        self.linear9 = nn.Linear(512*expansion, 2)
        self.linear10 = nn.Linear(512*expansion, 2)
        self.linear11 = nn.Linear(512*expansion, 2)

    def _make_layer(self, block, planes, num_blocks, expansion, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, expansion, stride))
            self.in_planes = planes * expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        outf = out.view(out.size(0), -1)
        outt1 = self.linear1(outf)
        outt2 = self.linear2(outf)
        outt3 = self.linear3(outf)
        outt4 = self.linear4(outf)
        outt5 = self.linear5(outf)
        outt6 = self.linear6(outf)
        outt7 = self.linear7(outf)
        outt8 = self.linear8(outf)
        outt9 = self.linear9(outf)
        outt10 = self.linear10(outf)
        outt11 = self.linear11(outf)

        return [outt1, outt2, outt3, outt4, outt5, outt6, outt7, outt8, outt9, outt10, outt11], outf, [out1, out2, out3, out4]

def ResNet18(num_classes = 10, expansion = 1):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, expansion)

class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()
        
        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0]*9, interm_dim)
        self.FC2 = nn.Linear(num_channels[1]*9, interm_dim)
        self.FC3 = nn.Linear(num_channels[2]*9, interm_dim)
        self.FC4 = nn.Linear(num_channels[3]*9, interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)
    
    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, f_filt=4):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.f_filt = f_filt
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, self.f_filt, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*6*6)),     #1024*14*12                            # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024*6*6, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(1024*6*6, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + 1, 1024*6*6),                           # B, 1024*8*8
            View((-1, 1024, 6, 6)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, self.f_filt, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 2, 2),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, r, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z,r],1)
        x_recon = self._decode(z)

        return  x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, r,z):  
        z = torch.cat([z, r], 1)
        return self.net(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0)) # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = criterion(diff,one)
    elif reduction == 'none':
        loss = criterion(diff,one)
    else:
        NotImplementedError()
    
    return loss

def test(models, epoch, method, criterion, dataloaders, args, mode='val'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    
    total = 0
    correct = [0]*args['NO_CLASSES']
    m = nn.LogSoftmax(dim=1)
    cm=[]
    for i in range(args['NO_CLASSES']):
        cm.append(np.zeros((2,2)))

    individual_predictor_loss = [[] for i in range(args['NO_CLASSES'])]
    total_lloss = 0.0
    total_loss = 0.0
    total_pred_loss = 0.0
        
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            scores, _, features = models['backbone'](inputs)
            labels = torch.reshape(labels, (labels.size(0), labels.size(2))).float()
            
            losses = []
            for i in range(args['NO_CLASSES']):
                s = m(scores[i])
                _,preds = torch.max(s, dim=1)

                cm[i]+=confusion_matrix(labels[:,i].cpu().numpy(), preds.cpu().numpy())
                correct[i]+=(preds == labels[:,i]).sum().item()  
                
                l = criterion[i](s.float(), labels[:,i].long())
                losses.append(l)
                individual_predictor_loss[i].append(l)    
                
            total += labels.size(0)
            target_loss=torch.stack(losses).mean(dim=0)
            
            if method == 'lloss' or 'TA-VAAL':
                if epoch > args['EPOCHL']:
                    features[0] = features[0].detach()
                    features[1] = features[1].detach()
                    features[2] = features[2].detach()
                    features[3] = features[3].detach()

                pred_loss = models['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=args['MARGIN'])

                total_lloss+= m_module_loss.item()

                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
                loss            = m_backbone_loss + args['WEIGHT'] * m_module_loss 
                total_loss+=loss.item()
            else:
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
                loss            = m_backbone_loss
    
    for i in range(args['NO_CLASSES']):
        individual_predictor_loss[i] = torch.stack(individual_predictor_loss[i]).mean().item()            
    
    return (100 * np.array(correct)) / total, cm, individual_predictor_loss, total_lloss, total_loss

iters = 0
def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, args): #epoch_loss, num_tasks):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models['backbone'].train()
    if method == 'lloss' or 'TA-VAAL':
        models['module'].train()
    global iters
    models['backbone'].train()
    models['module'].train()
    models['backbone'].to(device)
    models['module'].to(device)
    m = nn.LogSoftmax(dim=1)
    
    individual_predictor_loss = [[] for i in range(args['NO_CLASSES'])]
    total_lloss = 0.0
    total_loss = 0.0
    correct = [0 for i in range(args['NO_CLASSES'])]
    total=0

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        iters += 1

        optimizers['backbone'].zero_grad()
        if method == 'lloss' or 'TA-VAAL':
            optimizers['module'].zero_grad()
        
        scores, _, features = models['backbone'](inputs)
        
        labels = torch.reshape(labels, (labels.size(0), labels.size(2))).float()

        losses = []
        for i in range(args['NO_CLASSES']):
            s = m(scores[i])
            _,preds = torch.max(s, dim=1)
            correct[i]+=(preds == labels[:,i]).sum().item()  
            l = criterion[i](s.float(), labels[:,i].long())
            losses.append(l)
            individual_predictor_loss[i].append(l)
            
        total += labels.size(0)
        target_loss=torch.stack(losses).mean(dim=0)

        if method == 'lloss' or 'TA-VAAL':
            if epoch > args['EPOCHL']:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()
                
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=args['MARGIN'])
            
            total_lloss+= m_module_loss.item()

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss + args['WEIGHT'] * m_module_loss 
            total_loss+=loss.item()
        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss
            
        loss.backward()
        optimizers['backbone'].step()
        if method == 'lloss' or 'TA-VAAL':
            optimizers['module'].step()
            
    for i in range(args['NO_CLASSES']):
        individual_predictor_loss[i] = torch.stack(individual_predictor_loss[i]).mean().item()
            
    return loss, individual_predictor_loss, total_lloss, total_loss, (100 * np.array(correct)) / total

def train(models, method, criterion, optimizers, schedulers, dataloaders, cycle, args):
    
    print('>> Train a Model.')
    best_acc = 0.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    num_tasks=args['NO_CLASSES']
    
    rows = []
    
    print(len(dataloaders['train']))
    for epoch in range(args['no_of_epochs']):
        row = [cycle, epoch]

        best_loss = torch.tensor([0.5]).to(device)
        loss, individual_predictor_loss, total_lloss, total_loss, individual_acc = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, args)

        schedulers['backbone'].step()
        if method == 'lloss' or 'TA-VAAL':
            schedulers['module'].step()
            
        # logging individual_predictor_loss
        for i in range(num_tasks):
            args['writer-train'].add_scalar(str(cycle) + ' Individual predictor loss '+ str(i),
                    individual_predictor_loss[i], epoch)
        
        # logging total predictor loss
        args['writer-train'].add_scalar(str(cycle) + ' Total predictor loss ',
                    sum(individual_predictor_loss)/len(individual_predictor_loss), epoch)
        
        # logging total lloss
        args['writer-train'].add_scalar(str(cycle) + ' Total lloss ',
                    total_lloss/len(dataloaders['train']), epoch)
        
        # logging total loss
        args['writer-train'].add_scalar(str(cycle) + ' Total loss ',
                    total_loss/len(dataloaders['train']), epoch)
        
        
        ## logging predictor acc
        args['writer-train'].add_scalar(str(cycle) + ' Total predictor Acc ',
                    sum(individual_acc) / len(individual_acc), epoch)
        row.append(sum(individual_acc) / len(individual_acc))

        # logging individual_predictor_acc
        for i in range(num_tasks):
            args['writer-train'].add_scalar(str(cycle) + ' Individual predictor acc ' + str(i),
                        individual_acc[i], epoch)
            row.append(individual_acc[i])

        # Testing
        acc, cm, individual_predictor_loss, total_lloss, total_loss = test(models, epoch, method, criterion, dataloaders, args, mode='test')
        print('Epoch ' + str(epoch)+": "+'Mean Accuracy: ', acc.mean())
        
        
        # logging predictor_acc
        args['writer-val'].add_scalar(str(cycle) + ' Total predictor Acc ',
                    acc.mean(), epoch)
        row.append(acc.mean())

        # logging individual_predictor_acc
        for i in range(num_tasks):
            args['writer-val'].add_scalar(str(cycle) + ' Individual predictor acc ' + str(i),
                        acc[i], epoch)
            row.append(acc[i])
        
        # logging individual_predictor_loss
        for i in range(num_tasks):
            args['writer-val'].add_scalar(str(cycle) + ' Individual predictor loss '+ str(i),
                    individual_predictor_loss[i], epoch)
            
        # logging total predictor loss
        args['writer-val'].add_scalar(str(cycle) + ' Total predictor loss ',
                    sum(individual_predictor_loss)/len(individual_predictor_loss), epoch)
            
        # logging total lloss
        args['writer-val'].add_scalar(str(cycle) + ' Total lloss ',
                    total_lloss/len(dataloaders['test']), epoch)
        
        # logging total loss
        args['writer-val'].add_scalar(str(cycle) + ' Total loss ',
                    total_loss/len(dataloaders['test']), epoch)      
        
        rows.append(row)
        
    with open("results_" + args['group'] + '_' + str(args['attempt']) + ".csv", 'a') as csvfile: 
        csvwriter = csv.writer(csvfile) 

        # writing the data rows 
        csvwriter.writerows(rows)

            
#         if True and epoch % 20  == 0:
#             acc, cm = test(models, epoch, method, dataloaders, args, mode='test')
#             print('Mean Accuracy:', acc.mean())
#             for i in range(args['NO_CLASSES']):
#                 print('val_accuracy', i, ':', acc[i])
    print('>> Finished.')

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label,_ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle, args):
    
    vae = models['vae']
    discriminator = models['discriminator']
    task_model = models['backbone']
    ranker = models['module']
    
    task_model.eval()
    ranker.eval()
    vae.train()
    discriminator.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    vae = vae.to(device)
    discriminator = discriminator.to(device)
    task_model = task_model.to(device)
    ranker = ranker.to(device)

    adversary_param = 1
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 1

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = 1250 # int( (args['INCREMENTAL']*cycle+ args['SUBSET']) * args['EPOCHV'] / args['BATCH'] )
    print('Num of Iteration:', str(train_iterations))
    
    for iter_count in range(train_iterations):
        labeled_imgs, labels = next(labeled_data)
        unlabeled_imgs = next(unlabeled_data)[0]
        
        labeled_imgs = labeled_imgs.to(device)
        unlabeled_imgs = unlabeled_imgs.to(device)
        labels = labels.to(device)    
        
        if iter_count == 0 :
            r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0],1))).type(torch.FloatTensor).to(device)
            r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0],1))).type(torch.FloatTensor).to(device)
        else:
            with torch.no_grad():
                _,_,features_l = task_model(labeled_imgs)
                _,_,feature_u = task_model(unlabeled_imgs)
                r_l = ranker(features_l)
                r_u = ranker(feature_u)
        if iter_count == 0:
            r_l = r_l_0.detach()
            r_u = r_u_0.detach()
            r_l_s = r_l_0.detach()
            r_u_s = r_u_0.detach()
        else:
            r_l_s = torch.sigmoid(r_l).detach()
            r_u_s = torch.sigmoid(r_u).detach()   
            
        # VAE step
        for count in range(num_vae_steps): # num_vae_steps
            recon, _, mu, logvar = vae(r_l_s,labeled_imgs)
            unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
            unlab_recon, _, unlab_mu, unlab_logvar = vae(r_u_s,unlabeled_imgs)
            transductive_loss = vae_loss(unlabeled_imgs, 
                    unlab_recon, unlab_mu, unlab_logvar, beta)
        
            labeled_preds = discriminator(r_l,mu)
            unlabeled_preds = discriminator(r_u,unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                
            lab_real_preds = lab_real_preds.to(device)
            unlab_real_preds = unlab_real_preds.to(device)            

            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
            
            optimizers['vae'].zero_grad()
            total_vae_loss.backward()
            optimizers['vae'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_vae_steps - 1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]
                
                labeled_imgs = labeled_imgs.to(device)
                unlabeled_imgs = unlabeled_imgs.to(device)
                labels = labels.to(device)                

        # Discriminator step
        for count in range(num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = vae(r_l_s,labeled_imgs)
                _, _, unlab_mu, _ = vae(r_u_s,unlabeled_imgs)
            
            labeled_preds = discriminator(r_l,mu)
            unlabeled_preds = discriminator(r_u,unlab_mu)
            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            lab_real_preds = lab_real_preds.to(device)
            unlab_fake_preds = unlab_fake_preds.to(device)            
            
            dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                       bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

            optimizers['discriminator'].zero_grad()
            dsc_loss.backward()
            optimizers['discriminator'].step()

            # sample new batch if needed to train the adversarial network
            if count < (num_adv_steps-1):
                labeled_imgs, _ = next(labeled_data)
                unlabeled_imgs = next(unlabeled_data)[0]

                labeled_imgs = labeled_imgs.to(device)
                unlabeled_imgs = unlabeled_imgs.to(device)
                labels = labels.to(device)                
                
            if iter_count % 50 == 0:
                # print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))
                args['writer-train'].add_scalar(str(cycle) + ' Total VAE Loss ',
                        total_vae_loss.item(), iter_count)
                args['writer-train'].add_scalar(str(cycle) + ' Total DSC Loss ',
                        dsc_loss.item(), iter_count)
                
# Select the indices of the unlablled data according to the methods
def query_samples(model, method, data_unlabeled, subset, labeled_set, cycle, args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if method == 'TA-VAAL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=args['BATCH'], 
                                    sampler=SubsetSequentialSampler(subset), 
                                    pin_memory=True)
        labeled_loader = DataLoader(data_unlabeled, batch_size=args['BATCH'], 
                                    sampler=SubsetSequentialSampler(labeled_set), 
                                    pin_memory=True)
        
        vae = VAE()
        discriminator = Discriminator(32)
     
        models      = {'backbone': model['backbone'], 'module': model['module'], 'vae': vae, 'discriminator': discriminator}
        
        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator':optim_discriminator}

        train_vaal(models,optimizers, labeled_loader, unlabeled_loader, cycle+1, args)
        task_model = models['backbone']
        ranker = models['module']        
        all_preds, all_indices = [], []

        for images, _, indices in unlabeled_loader:                       
            images = images.to(device)
            with torch.no_grad():
                _,_,features = task_model(images)
                r = ranker(features)
                _, _, mu, _ = vae(torch.sigmoid(r),images)
                preds = discriminator(r,mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled
        _, arg = torch.sort(all_preds) 
        
        torch.save(vae, 'saved_history/models/vae-' + args['group'] +'cycle-'+str(cycle)+'.pth')
        torch.save(discriminator, 'saved_history/models/discriminator-' + args['group'] +'cycle-'+str(cycle)+'.pth')
        
    return arg


# Main
def main(args):
    
    targets = ['image_id', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat' ]

    if args['dataset'] == 'celeba':
        args['NO_CLASSES'] = len(targets) - 1
    
    method = args['method_type']
    results = open('results_'+str(args['method_type'])+"_"+args['dataset'] +'_main'+str(args['CYCLES'])+str(args['total'])+'.txt','w')
    print("Dataset: %s"%args['dataset'])
    print("Method type:%s"%method)
    
    if args['total']:
        args['TRIALS'] = 1
        args['CYCLES'] = 1
    else:
        args['CYCLES'] = args['CYCLES']
        
    # fields
    fields = ['cycle', 'epoch', 'total_train_pred_acc']
    for i in range(len(targets)-1):
        fields.append('train_pred_'+ str(i+1) + '_acc')
    fields.append('total_val_pred_acc')
    for i in range(len(targets)-1):
        fields.append('val_pred_'+ str(i+1) + '_acc')
    
    # name of csv file 
    filename = "results_" + args['group'] + '_' + str(args['attempt']) + ".csv"

    # writing to csv file 
    with open(filename, 'a') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerow(fields)     
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    for trial in range(args['trials']):
        
        # Load training and testing dataset
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args, targets)
        print('The entire datasize is {}'.format(len(data_train)))  
        
        ADDENDUM = adden
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)

        if args['total']:
            labeled_set= indices
        else:
            labeled_set = indices[:ADDENDUM]
            unlabeled_set = [x for x in indices if x not in labeled_set]
        
        train_loader = DataLoader(data_train, batch_size=args['BATCH'], 
                                     sampler=SubsetRandomSampler(labeled_set), 
                                     pin_memory=True, drop_last=True)
        test_loader  = DataLoader(data_test, batch_size=args['BATCH'],  drop_last=True)
        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        np.save("saved_history/initial-labelled-" + args['group'] + ".npy", np.array(labeled_set))
        print('Len: ', len(train_loader), 'Len:', len(test_loader))
        
        for cycle in range(args['CYCLES']):
            print(cycle)
            
            if not args['total']:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:args['SUBSET']]
                            
            resnet18    = ResNet18(num_classes=args['NO_CLASSES'], expansion=9).to(device)
            if method == 'lloss' or 'TA-VAAL':
                loss_module = LossNet().to(device)

            models      = {'backbone': resnet18}
            if method =='lloss' or 'TA-VAAL':
                models = {'backbone': resnet18, 'module': loss_module}
            torch.backends.cudnn.benchmark = True
                        
            labels_l = []
            for i in labeled_set:
                labels_l.append(data_train[i][1])

            counts = np.array(labels_l).reshape((-1, args['NO_CLASSES'])).sum(axis=0)

            wts_0 = (counts / len(labels_l))
            wts_1 = 1 - (counts / len(labels_l))
            print(wts_0)
            
            # Loss, criterion and scheduler (re)initialization
            criterion      = [nn.NLLLoss(reduction='none', weight = torch.tensor([wts_0[i], wts_1[i]]).to(device)).float() for i in range(args['NO_CLASSES'])]
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=args['LR'], 
                momentum=args['MOMENTUM'], weight_decay=args['WDECAY'])
 
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=args['MILESTONES'])
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            if method == 'lloss' or 'TA-VAAL':
                optim_module   = optim.SGD(models['module'].parameters(), lr=args['LR'], 
                    momentum=args['MOMENTUM'], weight_decay=args['WDECAY'])
                sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=args['MILESTONES'])
                optimizers = {'backbone': optim_backbone, 'module': optim_module}
                schedulers = {'backbone': sched_backbone, 'module': sched_module} 

            # Training and testing
            train(models, method, criterion, optimizers, schedulers, dataloaders, cycle+1, args)
            acc, cm, individual_predictor_loss, total_lloss, total_loss = test(models, args['no_of_epochs'], method, criterion, dataloaders, args, mode='test')
            torch.save(models['backbone'], 'saved_history/models/predictor-backbone-' + args['group'] + 'cycle-'+str(cycle+1)+'.pth')
            torch.save(models['module'], 'saved_history/models/predictor-module-'+args['group']+'cycle-'+str(cycle+1)+'.pth')
            
            print('Trial {}/{} || Cycle {}/{} || Label set size {}'.format(trial+1, args['trials'], cycle+1, args['CYCLES'], len(labeled_set)))

            for i in range(args['NO_CLASSES']):
                print('val_accuracy', i, ':', acc[i])
                print(cm[i])
                print("")
            
            if cycle == (args['CYCLES']-1):
                # Reached final training cycle
                print("Finished.")
                break
                
            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args)
            
            # Update the labeled dataset and the unlabeled dataset, respectively
            new_list = list(torch.tensor(subset)[arg][:args['INCREMENTAL']].numpy())
            labeled_set += list(torch.tensor(subset)[arg][-args['INCREMENTAL']:].numpy())
            listd = list(torch.tensor(subset)[arg][:-args['INCREMENTAL']].numpy()) 
            unlabeled_set = listd + unlabeled_set[args['SUBSET']:]
            print(len(labeled_set), min(labeled_set), max(labeled_set))
            
            np.save("saved_history/labelled-" + args['group'] + str(cycle) + ".npy", np.array(labeled_set))
            
            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=args['BATCH'], 
                                            sampler=SubsetRandomSampler(labeled_set), 
                                            pin_memory=True, drop_last=True)

    results.close()


splits = [0.1,0.15,0.2,0.25,0.3,0.35,0.4]

args = {
    'method_type': 'TA-VAAL', 
    'dataset': 'celeba',
    'group': 'head',
    'total': False,
    'trials': 1,
    'CYCLES': 7,
    'ADDENDUM': 16277, ## num of images increment
    'INCREMENTAL': 8138, ## 
    'BATCH': 32, ## 
    'SUBSET': 20000, ##
    'NO_CLASSES': 11, ##
    'LR': 1e-4,
    'WDECAY': 5e-4,
    'MOMENTUM': 0.9,
    'MILESTONES': [160, 240],
    'MARGIN': 1.0,
    'WEIGHT': 1.0,
    'no_of_epochs': 20,
    'lambda_loss': 1.2,
    's_margin': 0.1,
    'hidden_units': 128,
    'dropout_rate': 0.3,
    'EPOCHV':10, 
    'EPOCHL': 12, 
    'attempt': 1,
    'writer-train': SummaryWriter('logs/TAVAAL-CELEBA-head-1-Train'),
    'writer-val': SummaryWriter('logs/TAVAAL-CELEBA-head-1-Val')
}

if not os.path.isdir('logs'):
    os.makedirs('logs')

if not os.path.isdir('saved_history/models'):
    os.makedirs('saved_history/models')

main(args)
