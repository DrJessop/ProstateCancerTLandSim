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
from sklearn.metrics import confusion_matrix, f1_score, auc, roc_curve
import pandas as pd


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
                "{}/{}.nrrd".format(self.modality, index)
            )
            image = sitk.ReadImage(path)
            output = {"image": image}

        output["image"] = self.normalize.Execute(output["image"])
        output["image"] = sitk.GetArrayFromImage(output["image"])
        output["image"] = torch.from_numpy(output["image"]).float().to(self.device)
        return output


class CNN(nn.Module):
    """
    Baseline CNN with 5 Conv layers and 2 linear layers.
    """

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


def train_model(train_data, val_data, model, epochs, batch_size, optimizer, loss_function, show=False):
    """
    This function trains a model with batches of a given size, and if show=True, plots the loss, f1, and auc scores for
    the training and validation sets
    :param train_data: A dataloader containing batches of the training set
    :param val_data: A dataloader containing batches of the validation set
    :param model: The network being trained
    :param epochs: How many times the user wants the model trained on all of the training set data
    :param batch_size: How many data points are in a batch
    :param optimizer: Method used to update the network's weights
    :param loss_function: How the model will be evaluated
    :param show: Whether or not the user wants to see plots of loss, f1, and auc scores for the training and validation
    sets
    :return: None
    """

    print("The loss function being used is {}".format(loss_function))
    errors = []
    eval_errors = []
    f1_train = []
    auc_train = []
    f1_eval = []
    auc_eval = []
    num_training_batches = len(train_data)
    for epoch in range(epochs):
        print("Epoch {}".format(epoch + 1))
        print("Training mode")
        model.train()
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
        train_confusion_matrix = confusion_matrix(all_actual, all_preds).ravel()
        print("Training loss for epoch {} is {}".format(epoch + 1, train_loss_avg))
        print("Confusion matrix for epoch {}, tn={}, fp={}, fn={}, tp={}".format(epoch + 1, *train_confusion_matrix))
        f1_train.append(f1_score(all_actual, all_preds))
        print("F1 score for epoch {} is {}".format(epoch + 1, f1_train[-1]))
        fpr, tpr, _ = roc_curve(all_actual, all_preds, pos_label=1)
        auc_train.append(auc(fpr, tpr))
        print("AUC for epoch {} is {}\n".format(epoch + 1, auc_train[-1]))
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
        cm = confusion_matrix(all_actual, all_preds).ravel()
        print("Evaluation loss for epochs {} is {}".format(epoch, eval_loss_avg))
        print("Evaluation Confusion matrix for epoch {}, tn={}, fp={}, fn={}, tp={}".format(epoch + 1, *cm))
        f1_eval.append(f1_score(all_actual, all_preds))
        print("Evaluation F1 score for epoch {} is {}".format(epoch + 1, f1_eval[-1]))
        fpr, tpr, _ = roc_curve(all_actual, all_preds, pos_label=1)
        auc_eval.append(auc(fpr, tpr))
        print("Evaluation AUC for epoch {} is {}\n".format(epoch + 1, auc_eval[-1]))
        eval_errors.append(eval_loss_avg)
    if show:
        plt.plot(errors)
        plt.plot(eval_errors)
        plt.title("Training (blue) vs Cross-Validation (orange) Error (BCELoss)")
        plt.legend(["training loss", "validation loss"])
        plt.show()
        plt.plot(f1_train)
        plt.plot(f1_eval)
        plt.legend(["training f1", "validation f1"])
        plt.title("F1 Training vs F1 Cross-Validation")
        plt.show()
        plt.plot(auc_train)
        plt.plot(auc_eval)
        plt.legend(["training auc", "validation auc"])
        plt.title("AUC Training vs AUC Cross-Validation")
        plt.show()
    return


def test_predictions(dataloader, model, batch_size):
    """
    This function runs the model on the batches in the test set and returns a dataframe with ProxID, fid, and ClinSig
    columns. The predictions x <- ClinSig, 0 <= x <= 1, x <- R.
    :param dataloader: The data loader with the test batches
    :param model: The trained pytorch model
    :param batch_size: The size of a test batch
    :return: A dataframe as described above
    """

    predictions = pd.read_csv(r"/home/andrewg/PycharmProjects/assignments/ProstateX-TestLesionInformation/ProstateX-Findings-Test.csv")
    predictions.insert(4, "ClinSig", 0)
    predictions = predictions.drop(["pos", "zone"], axis=1)

    for idx, batch in enumerate(dataloader):
        outputs = model(batch["image"])
        start_batch = idx * batch_size
        end_batch = start_batch + batch_size
        predictions["ClinSig"].iloc[start_batch: end_batch] = outputs.flatten().tolist()

    return predictions


if __name__ == "__main__":
    # Define hyper-parameters
    batch_size = 4
    loss_function = nn.BCELoss()

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    image_folder_contents = os.listdir("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/adc")
    num_images = len(image_folder_contents)
    p_images = ProstateImages(modality="adc", train=True, device=device)

    num_train = int(np.round(0.8 * num_images))
    num_val = int(np.round(0.2 * num_images))
    training, validation = torch.utils.data.random_split(p_images, (num_train, num_val))
    dataloader_train = DataLoader(training, batch_size=batch_size, shuffle=False)
    dataloader_val = DataLoader(validation, batch_size=batch_size)

    # Model 1
    cnn = CNN()
    cnn.cuda()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    train_model(train_data=dataloader_train, val_data=dataloader_val, model=cnn, epochs=15,
                batch_size=batch_size, optimizer=optimizer, loss_function=loss_function, show=True)

    test_folder = "/home/andrewg/PycharmProjects/assignments/resampled_cropped/test/adc"
    p_images_test = ProstateImages(modality="adc", train=False, device=device)

    dataloader_test = DataLoader(p_images_test, batch_size=batch_size, shuffle=False)

    predictions = test_predictions(dataloader_test, cnn, batch_size)
    predictions.to_csv(r"/home/andrewg/PycharmProjects/assignments/predictions/preds.csv", index=False)


