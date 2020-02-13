import h5py
import numpy as np
import pathlib
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import tqdm

from tqdm import trange
from torch.autograd import Variable

from ssd.layers.modules import MultiBoxLoss
from ssd.generator import CalorimeterJetDataset
from ssd.net import build_ssd

def adjust_learning_rate(optimizer, gamma, lr):
    lr = lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def xavier(param):
    init.xavier_uniform_(param)


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

HOME = pathlib.Path().absolute()
DATA_SOURCE = '/eos/user/a/adpol/ceva/fast'

dataset = 'ssd-jet-3'
train_dataset_path = '%s/RSGraviton_NARROW_0.h5' % DATA_SOURCE
save_folder = os.path.join(HOME, 'models/')

# Learning Parameters
num_classes = 1
batch_size = 100
train_epochs = 50
num_workers = 1
lr = 1e-4
momentum = 0.9
gamma = 0.1
weight_decay = 5e-4
lr_steps = (30, 40, 50)

# Create a save directory
if not os.path.join(HOME, save_folder):
    os.mkdir(os.path.join(HOME, save_folder))

# Initialize Dataset
h5_train = h5py.File(train_dataset_path, 'r')
train_dataset = CalorimeterJetDataset(hdf5_dataset=h5_train)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           collate_fn=detection_collate,
                                           shuffle=True,
                                           num_workers=num_workers)

# Build SSD Network
ssd_net = build_ssd('train', 300, num_classes + 1, False)
net = ssd_net

# Data Parallelization
net = torch.nn.DataParallel(ssd_net)
cudnn.benchmark = True
net = net.cuda()

# Initialize Weights
ssd_net.vgg.apply(weights_init)
ssd_net.extras.apply(weights_init)
ssd_net.loc.apply(weights_init)
ssd_net.conf.apply(weights_init)

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Total network parameters: %s' % total_params)

optimizer = optim.SGD(net.parameters(),
                      lr=lr,
                      momentum=momentum,
                      weight_decay=weight_decay)

criterion = MultiBoxLoss(num_classes+1,
                         0.5,
                         True,
                         0,
                         True,
                         3,
                         0.5,
                         False,
                         True)

net.train()

for epoch in range(1, train_epochs+1):

    if epoch in lr_steps:
        adjust_learning_rate(optimizer, gamma, lr)

    batch_loss = [0]

    tr = trange(len(train_loader)*batch_size, file=sys.stdout)
    tr.set_description('Epoch {}'.format(epoch))

    for batch_index, (images, targets) in enumerate(train_loader):

        images = Variable(images.cuda())
        targets = [Variable(ann.cuda()) for ann in targets]

        optimizer.zero_grad()
        output = net(images)
        loss_l, loss_c = criterion(output, targets)
        loss = loss_l + loss_c
        loss.backward()
        batch_loss.append(loss.item())
        optimizer.step()

        if epoch < 10:
            tr.set_description('Epoch {}  {:.6f}'.format(epoch,
                    np.average(batch_loss)))
        else:
            tr.set_description('Epoch {} {:.6f}'.format(epoch,
                    np.average(batch_loss)))

        tr.update(len(images))

    tr.close()

    torch.save(ssd_net.state_dict(), save_folder + dataset + '.pth')

h5_train.close()

