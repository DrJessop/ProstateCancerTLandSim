import os
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader


def read_cropped_images(modality):
    """
    This function reads in images of a certain modality and stores them in the dictionary
    cropped_images
    :param modality: ex. t2, adc, bval, etc.
    :return: A dictionary where the first key is a patient number and the second key is the
    fiducial number (with 0 indexing)
    """
    cropped_images = {}
    destination = \
        r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/{}".format(modality)
    start_position_patient_number = -11
    end_position_patient_number = -8
    fiducial_number_pos = -6
    image_dir = os.listdir(destination)
    for image_file_name in image_dir:
        image = sitk.ReadImage("{}/{}".format(destination, image_file_name))
        patient_number = int(image_file_name[start_position_patient_number:
                                             end_position_patient_number + 1])
        fiducial_number = int(image_file_name[fiducial_number_pos])

        if patient_number not in cropped_images.keys():
            cropped_images[patient_number] = {}

        cropped_images[patient_number][fiducial_number] = image

    return cropped_images


class ProstateImages(Dataset):
    """
    This class's sole purpose is to provide the framework for fetching training/test data for the data loader which
    uses this class as a parameter
    """
    def __init__(self, modality, train, device):
        self.modality = modality
        self.train = train
        self.device = device
        self.normalize = sitk.NormalizeImageFilter()

    def __len__(self):
        if self.train:
            length = len(os.listdir("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/{}".format(
                                                                                                         self.modality))
                         )
        else:
            length = len(os.listdir("/home/andrewg/PycharmProjects/assignments/resampled_cropped/test/{}".format(
                                                                                                        self.modality))
                         )
        return length

    def __getitem__(self, index):
        if self.train:
            path = "{}/{}".format(
                "/home/andrewg/PycharmProjects/assignments/resampled_cropped/train",
                "{}/{}_{}.nrrd".format(self.modality, index, "{}")
            )
            try:
                final_path = path.format(0)
                image = sitk.ReadImage(final_path)
            except:
                final_path = path.format(1)
                image = sitk.ReadImage(final_path)

            # The last digit in the file name specifies cancer/non-cancer
            cancer_label = int(final_path.split('.')[0][-1])
            output = {"image": image, "cancer": cancer_label}

        else:
            path = "{}/{}".format(
                "/home/andrewg/PycharmProjects/assignments/resampled_cropped/test",
                "{}/{}.nrrd".format(self.modality)
            )
            image = sitk.ReadImage(path)
            output = {"image": image, "cancer": None}

        output["image"] = self.normalize.Execute(output["image"])
        output["image"] = sitk.GetArrayFromImage(output["image"])
        # print(output["image"].mean())
        output["image"] = torch.from_numpy(output["image"]).float().to(self.device)
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=1)
        self.dense1 = nn.Linear(in_features=1024, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=2)

    def forward_testing_method(self, data):
        print("Data going into the network is of shape {}".format(data.shape))
        data = data.unsqueeze(1)
        print("Data after reshaping is of shape {}".format(data.shape))
        data = self.conv1(data)
        print("Data after first convolution (3d) is of shape {}".format(data.shape))
        data = data.squeeze(2)
        print("Data after reshaping is of shape {}".format(data.shape))
        data = nn.BatchNorm2d(32).cuda()(data)
        data = nn.ReLU()(data)
        data = self.conv2(data)
        print("Data after second convolution (2d) is of shape {}".format(data.shape))
        data = nn.BatchNorm2d(32).cuda()(data)
        data = nn.ReLU()(data)
        data = self.max_pool1(data)
        print("Data after max pooling is of shape {}".format(data.shape))
        data = self.conv3(data)
        print("Data after third convolution (2d) is of shape {}".format(data.shape))
        data = nn.BatchNorm2d(64).cuda()(data)
        data = nn.ReLU()(data)
        data = self.conv4(data)
        print("Data after fourth convolution (2d) is of shape {}".format(data.shape))
        data = nn.BatchNorm2d(64).cuda()(data)
        data = nn.ReLU()(data)
        data = self.max_pool2(data)
        print("Data after max pooling is of shape {}".format(data.shape))
        data = self.conv5(data)
        print("Data after fifth convolution (2d) is of shape {}".format(data.shape))
        data = data.view(-1, 4 * 4 * 64)
        print("Data after reshaping is of shape {}".format(data.shape))
        data = self.dense1(data)
        print("Data after first dense layer is of shape {}".format(data.shape))
        data = nn.ReLU()(data)
        data = self.dense2(data)
        print("Data after second dense layer is of shape {}".format(data.shape))
        data = nn.ReLU()(data)
        data = nn.LogSoftmax(1)(data)
        print("Data after softmax is of shape {}".format(data.shape))
        return data

    def forward(self, data):
        data = data.unsqueeze(1)
        data = self.conv1(data)
        data = data.squeeze(2)
        data = nn.BatchNorm2d(32).cuda()(data)
        data = nn.ReLU()(data)
        data = self.conv2(data)
        data = nn.BatchNorm2d(32).cuda()(data)
        data = nn.ReLU()(data)
        data = self.max_pool1(data)
        data = self.conv3(data)
        data = nn.BatchNorm2d(64).cuda()(data)
        data = nn.ReLU()(data)
        data = self.conv4(data)
        data = nn.BatchNorm2d(64).cuda()(data)
        data = nn.ReLU()(data)
        data = self.max_pool2(data)
        data = self.conv5(data)
        data = data.view(-1, 4 * 4 * 64)
        data = self.dense1(data)
        data = nn.ReLU()(data)
        data = self.dense2(data)
        data = nn.ReLU()(data)
        data = nn.LogSoftmax(1)(data)
        return data


def train_model(train_data, val_data, model, epochs, optimizer, loss_function):
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        print("Training mode")
        model.train()
        num_training_batches = len(train_data)
        train_iter = iter(train_data)
        for batch_num in range(num_training_batches):
            batch = next(train_iter)
            images, class_vector = batch["image"], batch["cancer"]
            model.zero_grad()
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_function(preds, class_vector.cuda())
            loss.backward()
            optimizer.step()
            print("Batch {} error: {}".format(batch_num, loss))
        print("\nEvaluation mode")
        model.eval()
        num_val_batches = len(val_data)
        val_iter = iter(val_data)
        with torch.no_grad():
            for batch_num in range(num_val_batches):
                batch = next(val_iter)
                images, class_vector = batch["image"], batch["cancer"]
                preds = model(images)
                loss = loss_function(preds, class_vector.cuda())
                print("Eval error for batch {} is {}".format(batch_num, loss))
        print()
    return


if __name__ == "__main__":
    # Define hyper-parameters
    batch_size = 20
    optimizer = optim.Adam
    loss_function = nn.NLLLoss()

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    image_folder_contents = os.listdir("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/t2")
    num_images = len(image_folder_contents)
    p_images = ProstateImages(modality="t2", train=True, device=device)

    num_train = int(np.round(0.8 * num_images))
    num_val = int(np.round(0.2 * num_images))
    training, validation = torch.utils.data.random_split(p_images, (num_train, num_val))
    dataloader_train = DataLoader(training, batch_size=batch_size)
    dataloader_val = DataLoader(validation, batch_size=batch_size)

    cnn = CNN()
    cnn.cuda()
    optimizer = optimizer(cnn.parameters(), lr=0.00025)
    train_model(train_data=dataloader_train, val_data=dataloader_val, model=cnn, epochs=5,
                optimizer=optimizer, loss_function=loss_function)

