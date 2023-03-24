from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import shutil
from tqdm import trange
import time

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1, 1)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        x = F.relu(torch.add(x, y))
        return x
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.bn1 = nn.BatchNorm2d(8)

        self.bk1 = Residual(8, 16)
        self.bk2 = Residual(16, 16)

        self.bk3 = Residual(16, 32)
        self.bk4 = Residual(32, 32)

        self.bk5 = Residual(32, 64)
        self.bk6 = Residual(64, 64)

        self.bk7 = Residual(64, 128)
        self.bk8 = Residual(128, 128)

        self.fc = nn.Linear(36*128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.bk1(x)
        x = self.bk2(x)
        x = self.bk3(x)
        x = self.bk4(x)
        x = self.bk5(x)
        x = self.bk6(x)
        x = self.bk7(x)
        x = self.bk8(x)
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def main(args):

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu")

    train_loader, test_loader = dataset(args)

    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    File = 'LOSS' + str(args.batch_size)
    fileCreate(File)

    accuracy = np.zeros([args.epochs, 1])

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        accuracy[epoch-1] = test(model, device, test_loader)
        scheduler.step()
    end = time.perf_counter()
    # time used to train is 274

    print('The time used to train is ', round(end-start))

    saveData('accuracy.npy', accuracy, 'accuracy')

    if args.save_model is True:
        fileCreate('best_model')
        torch.save(model.state_dict(), "best_model/mnist_cnn.pt")

    return 0

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training (default: 512)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train (default: 15)')

    parser.add_argument('--num-workers', type=int, default=6, metavar='N', help='number of worker of torch to train (default: 10)')

    parser.add_argument('--lr', type=float, default=1, metavar='LR', help='learning rate (default: 1.0)')

    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--use-cuda', action='store_true', default=True, help='disables CUDA training')

    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')

    return parser.parse_args()

def dataset(args):

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if torch.cuda.is_available() and args.use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                        'pin_memory': True,
                        'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Make train dataset split
    trainset = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    # Make test dataset split
    testset = datasets.MNIST('data', train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    return train_loader, test_loader

def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    sum_up_batch_loss = 0
    lossFile = 'LOSS' + str(args.batch_size) + '/loss_epoch' + str(epoch) + '.npy'

    with trange(len(train_loader)) as pbar:
        for batch_idx, ((data, target), i) in enumerate(zip(train_loader, pbar)):
            pbar.set_description(f"epoch{epoch}/{args.epochs}")
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            sum_up_batch_loss += loss.cpu().detach().numpy()
            average_loss = sum_up_batch_loss/(batch_idx+1)
            pbar.set_postfix({'loss':'{:.4f}'.format(loss.cpu().detach().numpy()), 'average loss':'{:.4f}'.format(average_loss)})

            saveData(lossFile, loss.cpu().detach().numpy(), 'loss')

    return 0

def saveData(file, data, item_name):
    if os.path.exists(file) is True:
        dictionary = np.load(file, allow_pickle= True).item()
        data_temp = dictionary[item_name]
        data = np.append(data_temp, data)

    dictionary = {item_name: data}
    np.save(file, dictionary)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct_num = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            predict = output.argmax(dim=1, keepdim=True)
            correct_num += predict.eq(target.view_as(predict)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct_num / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct_num, len(test_loader.dataset), 100.*accuracy))

    return accuracy

def fileCreate(fileName):
    if os.path.exists(fileName) is True:
        shutil.rmtree(fileName)
        os.makedirs(fileName)
    else:
        os.makedirs(fileName)


if __name__ == '__main__':
    args = parse_args()
    main(args)