from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
import random


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

    device = torch.device("cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu")

    model = Net().to(device)
    model.load_state_dict(torch.load("best_model/mnist_cnn.pt"))

    fileCreate('pred_res')
    fileCreate('pred_res/correct')
    fileCreate('pred_res/false')

    test_loader = dataset(args)

    test(model, device, test_loader)

    return 0

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N', help='input batch size for testing (default: 1000)')

    parser.add_argument('--num-workers', type=int, default=4, metavar='N', help='number of worker of torch to train (default: 4)')

    parser.add_argument('--use-cuda', action='store_true', default=True, help='disables CUDA training')

    return parser.parse_args()

def dataset(args):

    test_kwargs = {'batch_size': args.test_batch_size}
    if torch.cuda.is_available() and args.use_cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                        'pin_memory': True,
                        'shuffle': True}
    test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    testset = datasets.MNIST('data', train=False,
                       transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    return test_loader

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct_num = 0

    with torch.no_grad():
        num_correct_img = 1
        num_false_img = 1
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            predict = output.argmax(dim=1, keepdim=True)
            target = target.view_as(predict)
            correct_num += predict.eq(target).sum().item()
            for i, ((pred, targ), image_array) in enumerate(zip(zip(predict, target), data)):
                if pred == targ:
                    if random.randint(1,750) == 10:
                        fileName = 'pred_res/correct/c_' + str(num_correct_img) + '_targ_' + str(targ.item()) + '_pred' + str(pred.item()) + '_con_{:.4f}.png'.format(F.softmax(output,dim=1).cpu().numpy()[i, pred])
                        num_correct_img += 1
                        drawImage(fileName, image_array.reshape(28,28))
                else:
                    if random.randint(1,10) == 10:
                        fileName = 'pred_res/false/f_' + str(num_false_img) + '_targ_' + str(targ.item()) + '_pred' + str(pred.item()) + '_con_{:.4f}.png'.format(F.softmax(output,dim=1).cpu().numpy()[i, pred])
                        num_false_img += 1
                        drawImage(fileName, image_array.reshape(28,28))
    test_loss /= len(test_loader.dataset)
    accuracy = correct_num / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct_num, len(test_loader.dataset), 100.*accuracy))

    return 0

def fileCreate(fileName):
    if os.path.exists(fileName) is True:
        shutil.rmtree(fileName)
        os.makedirs(fileName)
    else:
        os.makedirs(fileName)

def drawImage(fileName, image_array):
    img = np.array(image_array.to("cpu"))
    img = 255 * (img * 0.3081 + 0.1307)
    img = img.astype(np.uint8)
    cv2.imwrite(fileName,img)

    return 0       

if __name__ == '__main__':
    args = parse_args()
    main(args)