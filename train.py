'''
adapted from https://github.com/fungtion/DANN_py3/blob/master/main.py
by James Kim
May 25, 2023
'''

import random
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import DDAModel
from eval import eval

def train():
    source_dataset_name = 'MNIST'
    target_dataset_name = 'mnist_m'
    source_image_root = os.path.join('repo/dataset', source_dataset_name)
    target_image_root = os.path.join('repo/dataset', target_dataset_name)
    model_root = 'repo/models'
    cuda = True
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 128
    image_size = 28
    n_epoch = 100

    manual_seed  = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    #load data
    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,),std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    dataset_source = datasets.MNIST(
        root='dataset',
        train=True,
        transform=img_transform_source,
        download=True
    )

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8 # CHECK I think there is way to get num_workers for current environment check U-Net implementation
    )

    train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

    dataset_target = GetLoader(
        data_root=os.path.join(target_image_root, 'mnist_m_train'),
        data_list=train_list,
        transform=img_transform_target
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    # load model
    model = DDAModel()

    # setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr) #Adam in original code

    loss_label = torch.nn.NLLLoss() # CHECK is it the right Loss function?
    loss_domain = torch.nn.NLLLoss() # CHECK should be BCE, but getting: ValueError: Target size (torch.Size([128])) must be the same as input size (torch.Size([128, 2]))

    if cuda:
        model = model.cuda()
        loss_label = loss_label.cuda()
        loss_domain = loss_domain.cuda()

    for p in model.parameters():
        p.requires_grad = True

    # training
    best_acc_t = 0.
    gamma = 10
    for epoch in range(n_epoch):
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)
        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            lamda = 2. / (1. + np.exp(-gamma * p))-1

            # training model using source data
            data_source = next(data_source_iter)
            s_img, s_label = data_source

            model.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                domain_label = domain_label.cuda()
            
            label_output, domain_output = model(input_data=s_img, lamda=lamda)
            loss_s_label = loss_label(label_output, s_label)
            loss_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = next(data_target_iter)
            t_img, _ = data_target

            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()

            if cuda:
                t_img = t_img.cuda()
                domain_label = domain_label.cuda()

            _, domain_output = model(input_data=t_img, lamda=lamda)
            loss_t_domain = loss_domain(domain_output, domain_label)
            loss = loss_t_domain + loss_s_domain + loss_s_label
            loss.backward()
            optimizer.step()

            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], loss_s_label: %f, loss_s_domain: %f, loss_t_domain: %f' \
                            % (epoch, i+1, len_dataloader, loss_s_label.data.cpu().numpy(),
                                loss_s_domain.data.cpu().numpy(), loss_t_domain.data.cpu().item()))
            sys.stdout.flush()
            torch.save(model, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))

            print('\n')
            acc_s = eval(source_dataset_name)
            print('Accuracy of the %s dataset: %f' % ('mnist', acc_s))
            acc_t = eval(target_dataset_name)
            print('Accuracy of the %s dataset: %f' % ('mnist_m', acc_t))
            if acc_t > best_acc_t:
                best_acc_s = acc_s
                best_acc_t = acc_t
                torch.save(model, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))
        print('============ Summary ============= \n')
        print('Accuracy of the %s dataset: %f' % ('mnist', best_acc_s))
        print('Accuracy of the %s dataset: %f' % ('mnist_m', best_acc_t))
        print('Corresponding model was save in ' + model_root + '/mnist_mnistm_model_epoch_best.pth')