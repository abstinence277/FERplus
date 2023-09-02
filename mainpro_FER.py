'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import csv

from torchtoolbox.transform import Cutout
from torchvision.models import AlexNet

import utils
from fer import FER2013
from models import *
import matplotlib

from models.Resnet50 import resnet50, resnet101, resnet18
from models.rsnet import  rsnet18

matplotlib.use('agg')
import matplotlib.pyplot as plt


#定义两个数组
Loss_train_list = []
Accuracy_train_list = []
Loss_publictest_list = []
Accuracy_publictest_list = []
Loss_privatetest_list = []
Accuracy_privatetest_list = []

#创建解析器并添加、解析参数
parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()

#store dato to csv
f = open('loss_accuracy_rsnet.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)


#初始化值
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9
cut_size = 44
#total_epoch = 250
total_epoch = 250
path = os.path.join(opt.dataset + '_' + opt.model)
# Data
print('==> Preparing data..')

#-----------------图片预处理（train/test）
#torchvision的输出是[0,1]的PILImage图像,我们把它转换为归一化范围为[-1, 1]的张量
transform_train = transforms.Compose([
    transforms.RandomCrop(44),#裁剪
    transforms.RandomHorizontalFlip(),#按照概率p对PIL图片进行水平翻转，p=0.5
    transforms.ToTensor(),#转为张量，同时归一化
])
#在图像的上下左右以及中心裁剪出尺寸为size的5张图片，在对对这5张图片进行水平或垂直镜像获得10张图片
transform_test = transforms.Compose([
    transforms.TenCrop(44),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split = 'Training', transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
#PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
#PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False)

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model  == 'VGG13':
    net = VGG('VGG13')
elif opt.model  == 'Resnet18':
    net = resnet18()
elif opt.model  == 'Resnet50':
    net = resnet50()
elif opt.model  == 'Resnet101':
    net = resnet101()
elif opt.model  == 'Rsnet18':
    net = rsnet18()
elif opt.model  == 'AlexNet':
    net = AlexNet()

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=1, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
#模型定义完之后，意味着给出输入，就可以得到输出的结果。那么就来比较 outputs 和 targets 之间的区别，那么就需要用到损失函数来描述
#criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([4.13,68.40,16.27,1.22,2.94,2.78,1.00])).float(),size_average=True)
#criterion.cuda()
#criterion = FocalLoss(alpha=torch.from_numpy(np.array([0.24,0.01,0.06,0.82,0.34,0.36,1.00])).float(),class_num=7)
#criterion = FocalLoss(class_num=7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac # 衰减系数
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    output_loss = train_loss / (batch_idx + 1)
    output_acc = 100. * correct / total
    Loss_train_list.append(output_loss)
    Accuracy_train_list.append(output_acc)

    Train_acc = 100.*correct/total

def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(outputs_avg, targets)
            PublicTest_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    PublicTest_acc = 100.*correct/total

    loss_publictest=PublicTest_loss / (batch_idx + 1)
    acc_publictest=100. * correct / total
    Loss_publictest_list.append(loss_publictest)
    Accuracy_publictest_list.append(acc_publictest)

    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch

def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(outputs_avg, targets)
            PrivateTest_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    PrivateTest_acc = 100.*correct/total

    loss_prvatetest=PrivateTest_loss / (batch_idx + 1)
    acc_privatetest=100. * correct / total
    Loss_privatetest_list.append(loss_prvatetest)
    Accuracy_privatetest_list.append(acc_privatetest)

    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
	        'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
    	    'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

for epoch in range(start_epoch, total_epoch):
    train(epoch)
    PublicTest(epoch)
    PrivateTest(epoch)


csv_writer.writerow(Accuracy_train_list)
csv_writer.writerow(Loss_train_list)
csv_writer.writerow(Accuracy_publictest_list)
csv_writer.writerow(Loss_publictest_list)
csv_writer.writerow(Accuracy_privatetest_list)
csv_writer.writerow(Loss_privatetest_list)
plt.subplot(2, 1, 1)
f.close()

print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)


