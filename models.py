import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch


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
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=1)
        self.gap = nn.AvgPool2d(kernel_size=(4, 4))
        self.dense = nn.Linear(in_features=64, out_features=2)
        self.cuda_destination = cuda_destination

    def forward(self, data, want_activation_maps=False):
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
        activation_maps = self.conv5(data)
        activation_maps = nn.ReLU()(activation_maps)
        data = self.gap(activation_maps)
        data = data.view(-1, 64)  # Vector representation
        data = self.dense(data)
        data = nn.Softmax(1)(data)
        if want_activation_maps:
            return data, activation_maps
        return data

    def class_activation_mapping(self, image):
        image = image.unsqueeze(0)
        parameters = list(self.named_parameters())[-4:]
        bias_vector = parameters[-1][1][:]
        weights = parameters[-2][1][:]

        outputs, activation_maps = self.forward(image, want_activation_maps=True)
        outputs = [float(output) for output in outputs[0]]

        activation_maps = activation_maps.squeeze(0).view(64, 4 * 4)
        class_activations = weights.mm(activation_maps).view(2, 4, 4)
        for idx in range(len(class_activations)):
            class_activations[idx] = class_activations[idx] + bias_vector[idx]

        class_activations = class_activations.unsqueeze(0).unsqueeze(2).squeeze(0)

        class_activations = nn.Upsample(size=(32, 32), mode="bilinear")(class_activations)
        if outputs[0] < outputs[1]:
            class_activations = torch.cat((class_activations[1], class_activations[0]), 0)
            outputs = [("cancer", outputs[1]), ("non-cancer", outputs[0])]
        else:
            outputs = [("non-cancer", outputs[0]), ("cancer", outputs[1])]

        # Returns a 2 x 32 x 32 tensor and what each class activation map represents
        return class_activations.squeeze(0).squeeze(1), outputs

    @staticmethod
    def visualize(image, heatmap, code=0, alpha=0.4):

        image = image[1].cpu().detach().numpy()
        heatmap = heatmap[0][code].cpu().detach().numpy()

        plt.imshow(image, cmap="gray", interpolation="bilinear")
        plt.imshow(heatmap, cmap="jet", alpha=alpha, interpolation="bilinear")
        plt.axis("off")
        plt.colorbar()
        plt.show()


class MNIST_CNN(nn.Module):

    def __init__(self, cuda_destination):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1)
        self.gap = nn.AvgPool2d(kernel_size=(4, 4))
        self.dense = nn.Linear(in_features=64, out_features=10)
        self.cuda_destination = cuda_destination

    def forward(self, data, want_activation_maps=False):
        data = data.unsqueeze(1)
        data = self.conv1(data)
        data = data.squeeze(2)
        data = nn.ReLU()(data)
        data = self.conv3(data)
        data = nn.ReLU()(data)
        activation_maps = self.conv4(data)
        activation_maps = nn.ReLU()(activation_maps)
        data = self.gap(activation_maps)
        data = data.view(-1, 64)  # Vector representation
        data = self.dense(data)
        data = nn.LogSoftmax(1)(data)
        if want_activation_maps:
            return data, activation_maps

        return data

    def class_activation_mapping(self, image):
        parameters = list(self.named_parameters())[-4:]
        bias_vector = parameters[-1][1][:]
        weights = parameters[-2][1][:]

        outputs, activation_maps = self.forward(image, want_activation_maps=True)
        outputs = [float(output) for output in outputs[0]]

        activation_maps = activation_maps.squeeze(0).view(64, 4 * 4)
        class_activations = weights.mm(activation_maps).view(10, 4, 4)
        for idx in range(len(class_activations)):
            class_activations[idx] = class_activations[idx] + bias_vector[idx]

        class_activations = class_activations.unsqueeze(0).unsqueeze(2).squeeze(0)

        # plt.imshow(class_activations)
        # plt.show()
        probs = [np.power(np.e, output) for output in outputs]
        argmax = torch.argmax(torch.tensor(probs))
        # print(probs)
        class_activations = nn.Upsample(size=(28, 28), mode="bilinear")(class_activations)
        class_activations = class_activations[torch.argmax(torch.tensor(probs))]
        # Returns a 2 x 32 x 32 tensor and what each class activation map represents
        return class_activations.squeeze(0), outputs

    @staticmethod
    def visualize(image, heatmap, alpha=0.4):
        image = image.cpu().detach().numpy()[0]
        heatmap = heatmap[0].cpu().detach().numpy()

        plt.imshow(image, cmap="gray")
        plt.imshow(heatmap, cmap="jet", alpha=alpha)
        plt.colorbar()
        plt.show()

