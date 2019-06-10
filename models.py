import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, cuda_destination):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=1)
        self.dense1 = nn.Linear(in_features=1024, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=1)

        self.cuda_destination = cuda_destination

    def forward(self, data):
        data = data.unsqueeze(1)
        data = self.conv1(data)
        data = data.squeeze(2)
        data = nn.ReLU()(data)
        data = self.conv2(data)
        data = nn.BatchNorm2d(32).cuda(self.cuda_destination)(data)
        data = nn.ReLU()(data)
        data = self.max_pool1(data)
        data = self.conv3(data)
        data = nn.BatchNorm2d(64).cuda(self.cuda_destination)(data)
        data = nn.ReLU()(data)
        data = self.conv4(data)
        data = nn.ReLU()(data)
        data = self.max_pool2(data)
        data = self.conv5(data)
        data = nn.ReLU()(data)
        data = data.view(-1, 4 * 4 * 64)
        data = self.dense1(data)
        data = nn.ReLU()(data)
        data = self.dense2(data)
        data = nn.Sigmoid()(data)
        return data


class CNN2(nn.Module):
    def __init__(self, cuda_destination):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(2, 2, 2), stride=2, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(2, 2), stride=1)
        self.linear1 = nn.Linear(in_features=288, out_features=100)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(in_features=100, out_features=1)

        self.cuda_destination = cuda_destination

    def forward(self, data):
        data = data.unsqueeze(1)
        data = self.conv1(data)
        data = nn.ELU()(data)
        data = nn.ELU()(data)
        data = nn.BatchNorm3d(32).cuda(self.cuda_destination)(data)
        data = self.pool1(data)
        data = data.squeeze(2)
        data = self.conv2(data)
        data = nn.ELU()(data)
        data = nn.ELU()(data)
        data = self.conv3(data)
        data = nn.ELU()(data)
        data = nn.ELU()(data)
        data = data.view(-1, 8 * 6 * 6)
        data = self.linear1(data)
        data = nn.ELU()(data)
        data = nn.ELU()(data)
        data = self.dropout(data)
        data = self.linear2(data)
        data = nn.Sigmoid()(data)
        return data