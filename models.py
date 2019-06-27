import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def image_visualize(data, feature_maps):

    data = data.squeeze(0).cpu().detach().numpy()
    grid_length = 6
    grid_height = int(np.ceil(feature_maps / grid_length))
    grid_plot, _ = plt.subplots(grid_height, grid_length)

    last_n_to_remove = grid_length * grid_height - feature_maps
    for del_idx in range(last_n_to_remove):
        grid_plot.delaxes(grid_plot.axes.pop())
    for feature_map in range(feature_maps):
        grid_plot.axes[feature_map].imshow(data[feature_map])
        grid_plot.axes[feature_map].axis("off")
    plt.show()


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

    def forward(self, data, show_data=None):
        if show_data == 0:
            plt.imshow(data.squeeze(0).cpu().detach().numpy()[1])
            plt.show()
        data = data.unsqueeze(1)
        data = self.conv1(data)
        data = data.squeeze(2)
        data = nn.ReLU()(data)
        if show_data == 1:
            image_visualize(data, self.conv1.out_channels)
        data = self.conv2(data)
        data = nn.BatchNorm2d(32).cuda(self.cuda_destination)(data)
        data = nn.ReLU()(data)
        if show_data == 2:
            image_visualize(data, self.conv2.out_channels)
        data = self.max_pool1(data)
        if show_data == 3:
            num_feature_maps, _, _ = data.squeeze(0).cpu().detach().numpy()
            image_visualize(data, num_feature_maps)
        data = self.conv3(data)
        data = nn.BatchNorm2d(64).cuda(self.cuda_destination)(data)
        data = nn.ReLU()(data)
        if show_data == 4:
            image_visualize(data, self.conv3.out_channels)
        data = self.conv4(data)
        data = nn.ReLU()(data)
        if show_data == 5:
            image_visualize(data, self.conv4.out_channels)
        data = self.max_pool2(data)
        if show_data == 6:
            num_feature_maps, _, _ = data.squeeze(0).cpu().detach().numpy()
            image_visualize(data, num_feature_maps)
        data = self.conv5(data)
        data = nn.ReLU()(data)
        if show_data == 7:
            image_visualize(data, self.conv5.out_channels)
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
