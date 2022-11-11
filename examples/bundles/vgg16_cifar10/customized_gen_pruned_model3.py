import torch
import time
from torchvision import datasets, transforms
from pruned_vgg_maxpool import VGG
from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchsummary import summary
from thop import profile

import os
import random
import torch.onnx
import onnxruntime
import numpy as np
import argparse
import _pickle as cPickle


def get_data(dataset, data_dir, batch_size, test_batch_size):
    '''
    get data
    '''
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    if dataset == 'cifar10':
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True, **kwargs)
        
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()


    return train_loader, val_loader, criterion


#def get_vgg_model(pretrained_model_dir, my_shape=None, device):
#    model = VGG(depth=16).to(device)
#    model.load_state_dict(torch.load(pretrained_model_dir))
#    return model

def test2(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_time = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            t0 = time.time()
            output = model(data)
            t1 = time.time()
            total_time += (t1-t0)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    fps = len(val_loader.dataset) / total_time
    print('\nPyTorch Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), FPS: {:.2f}\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy, fps))

    return accuracy, fps

def train( model, device, train_loader, criterion, optimizer, epoch, callback=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def get_trained_model_optimizer(epochs, device, train_loader, val_loader, criterion, depth=16, save_name=None) :
    model = VGG(depth=depth).to(device)
#    model = ResNet50().to(device)
    summary(model, (3, 32, 32))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(
#        optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1)
        optimizer, milestones=[5, 10], gamma=0.1)
    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        train( model, device, train_loader, criterion, optimizer, epoch)
        scheduler.step()
        acc, _ = test2(model, device, criterion, val_loader)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            state_dict = model.state_dict()
    model.load_state_dict(state_dict)
    print('Best acc:', best_acc)
    print('Best epoch:', best_epoch)

    if save_name is not None:
        torch.save(state_dict, save_name)
    return model, optimizer, best_acc

def count_cifar_ops(model, device):
    input = torch.randn(1,3,32,32)
    macs, params = profile(model.to(device), inputs=(input,))
    return macs, params

def main(args):
    data_dir = args.data_dir
    dataset = args.dataset
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    pretrain_epochs = args.pretrain_epochs
    output_data_dir = args.output_data_dir

    num_output = 16
    f_dir = 612
#    for num_output in range(86,100):
#        my_shape = cPickle.load(open(os.path.join('/home/taeho/github/evta2/output', str(num_output), 'my_shape.p'),'rb'))

    train_loader, val_loader, criterion = get_data(dataset, data_dir, batch_size, test_batch_size)
#    torch.save(val_loader, 'val_loader.bin')
#    return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    device = torch.device("cpu")

    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    if not os.path.exists(os.path.join(output_data_dir, str(num_output))):
        os.makedirs(os.path.join(output_data_dir, str(num_output)))
    model, optimizer, acc = get_trained_model_optimizer(pretrain_epochs, device, train_loader, val_loader, criterion, num_output, os.path.join(output_data_dir, str(f_dir), 'model_trained.pth'))
#        cPickle.dump(my_shape, open(os.path.join(output_data_dir, str(num_output),'my_shape.p'),'wb'))
#        device_cpu = torch.device('cpu')
#        model_cpu = VGG(my_shape = my_shape, depth=16)
#        model_cpu.load_state_dict(torch.load(os.path.join(output_data_dir, str(num_output), 'model_trained.pth'), map_location=device_cpu))

#        macs, params = count_cifar_ops(model_cpu)
#    macs, params = count_cifar_ops(model, device)
#    perf_file = open(os.path.join(output_data_dir, str(num_output), 'perf.txt'), 'w')
#    perf_file.write("Accuracy : {}, MACs : {}, Params : {}".format(acc, macs, params))
#    perf_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pruned model generator')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--pretrain-epochs', type=int, default=100,
                        help='number of epochs to pretrain the model')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--output-data-dir', type=str, default='./output',
                        help='For saving output data')
    args = parser.parse_args()
    main(args)
