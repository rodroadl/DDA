'''
adapted from https://github.com/fungtion/DANN_py3/blob/master/test.py
by James Kim
May 25, 2023
'''

import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets

def eval(dataset_name,root=''):
    assert dataset_name in ['MNIST', 'mnist_m']

    model_root = root+'models'
    image_root = os.path.join(root+'dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    lamda = 0

    """load data"""
    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,),std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
    ])

    if dataset_name == 'mnist_m':
        test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')

        dataset = GetLoader(
            data_root=os.path.join(image_root, 'mnist_m_test'),
            data_list=test_list,
            transform=img_transform_target
        )
    else:
        dataset = datasets.MNIST(
            root='dataset',
            train=False,
            transform=img_transform_source
        )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """test"""

    model = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_current.pth'
    ))
    model = model.eval()

    if cuda: model = model.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    n_total = 0
    n_correct = 0

    for i in range(len_dataloader):
        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
        
        label_output, _ = model(input_data=t_img, lamda=lamda)
        pred = label_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    acc = n_correct.data.numpy() * 1.0 / n_total

    return acc
