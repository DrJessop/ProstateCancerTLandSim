import os
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, auc


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
        self.dense2 = nn.Linear(in_features=256, out_features=1)

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
        # data = nn.ReLU()(data)
        data = nn.Sigmoid()(data)
        return data


class CNNSimple(nn.Module):
    def __init__(self):
        super(CNNSimple, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, 1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lin1 = nn.Linear(576, 288)
        self.lin2 = nn.Linear(288, 1)

    def forward(self, data):
        data = data.unsqueeze(1)
        data = self.conv1(data)
        data = data.squeeze(2)
        data = self.max_pool1(data)
        data = nn.ReLU()(data)
        data = self.conv2(data)
        data = self.max_pool2(data)
        data = nn.ReLU()(data)
        data = data.view(-1, 16*6*6)
        data = self.lin1(data)
        data = nn.ReLU()(data)
        data = self.lin2(data)
        data = nn.Sigmoid()(data)
        return data


def train_model(train_data, val_data, model, epochs, optimizer, loss_function, show=False):
    global batch_size
    print(loss_function)
    errors = []
    eval_errors = []
    f1_train = []
    f1_eval = []
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        print("Training mode")
        model.train()
        num_training_batches = len(train_data)
        train_iter = iter(train_data)
        model.zero_grad()
        train_loss = 0
        all_preds = []
        all_actual = []
        for batch_num in range(num_training_batches):
            batch = next(train_iter)
            images, class_vector = batch["image"], batch["cancer"].float().cuda().unsqueeze(1)
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_function(preds, class_vector)
            hard_preds = torch.round(preds)
            all_preds.extend(hard_preds.squeeze(-1).tolist())
            all_actual.extend(class_vector.squeeze(-1).tolist())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss_avg = train_loss / (num_training_batches*batch_size)
        print("Training loss for epoch {} is {}".format(epoch, train_loss_avg))
        print("Confusion matrix for epoch {}, tn={}, fp={}, fn={}, tp={}".format(epoch, *confusion_matrix(all_actual,
                                                                                 all_preds).ravel()))
        f1_train.append(f1_score(all_actual, all_preds))
        print("F1 score for epoch {} is {}\n".format(epoch, f1_train[-1]))
        errors.append(train_loss_avg)
        print("Evaluation mode")
        model.eval()
        num_val_batches = len(val_data)
        val_iter = iter(val_data)
        eval_loss = 0
        all_preds = []
        all_actual = []
        with torch.no_grad():
            for batch_num in range(num_val_batches):
                batch = next(val_iter)
                images, class_vector = batch["image"], batch["cancer"].float().cuda().unsqueeze(1)
                preds = model(images)
                loss = loss_function(preds, class_vector)
                eval_loss += loss.item()
                hard_preds = torch.round(preds)
                all_preds.extend(hard_preds.squeeze(-1).tolist())
                all_actual.extend(class_vector.squeeze(-1).tolist())
        eval_loss_avg = eval_loss / (num_val_batches*batch_size)
        print("Evaluation loss for epochs {} is {}".format(epoch, eval_loss_avg))
        print("Evaluation Confusion matrix for epoch {}, tn={}, fp={}, fn={}, tp={}".format(epoch,
                                                                                            *confusion_matrix(all_actual,
                                                                                            all_preds).ravel()))
        f1_eval.append(f1_score(all_actual, all_preds))
        print("Evaluation F1 score for epoch {} is {}".format(epoch, f1_eval[-1]))
        eval_errors.append(eval_loss_avg)
        print()
    if show:
        plt.plot(errors)
        plt.plot(eval_errors)
        plt.title("Training (blue) vs Cross-Validation (orange) Error (BCELoss)")
        plt.legend(["training loss", "validation loss"])
        plt.show()
        input("Press enter to continue...")
        plt.plot(f1_train)
        plt.plot(f1_eval)
        plt.legend(["training f1", "validation f1"])
        plt.title("F1 Training vs F1 Cross-Validation")
        plt.show()
    return


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    # Define hyper-parameters
    batch_size = 4
    optimizer = optim.Adam
    loss_function = nn.BCELoss()

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    image_folder_contents = os.listdir("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/adc")
    num_images = len(image_folder_contents)
    p_images = ProstateImages(modality="adc", train=True, device=device)

    num_train = int(np.round(0.8 * num_images))
    num_val = int(np.round(0.2 * num_images))
    training, validation = torch.utils.data.random_split(p_images, (num_train, num_val))
    dataloader_train = DataLoader(training, batch_size=batch_size)
    dataloader_val = DataLoader(validation, batch_size=batch_size)

    # Model 1
    cnn = CNN()
    cnn.cuda()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    train_model(train_data=dataloader_train, val_data=dataloader_val, model=cnn, epochs=20,
                optimizer=optimizer, loss_function=loss_function, show=True)

    # Model 2
    cnn2 = CNNSimple()
    cnn2.cuda()
    cnn2(next(iter(dataloader_train))["image"])
    # optimizer = optim.Adam(cnn2.parameters(), lr=0.005)
    optimizer = optim.SGD(cnn2.parameters(), lr=0.001, momentum=0.9)
    # train_model(train_data=dataloader_train, val_data=dataloader_val, model=cnn2, epochs=20,
    #             optimizer=optimizer, loss_function=loss_function, show=True)