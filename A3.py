import os
import SimpleITK as sitk
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, auc, roc_curve
import pandas as pd
import pickle as pk


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

    def __init__(self, modality, train, device, mapping=None):
        self.modality = modality
        self.train = train
        self.device = device
        self.normalize = sitk.NormalizeImageFilter()
        self.num_images = num_images
        if self.train:
            self.mapping = mapping
            self.map_num = 0
            # The 0th index may vary depending on the first key of the hash function
            self.first_index = sorted(self.mapping[self.map_num])[0]
            self.length = len(self.mapping[self.map_num])
        else:
            path = "/home/andrewg/PycharmProjects/assignments/resampled_cropped/test/{}/".format(self.modality)
            sorted_path = sorted(os.listdir(path))
            self.length = len(sorted_path)
            self.first_index = int(sorted_path[0].split('.')[0])

    def __len__(self):
        return self.length

    def change_map_num(self, new_map_num):
        self.map_num = new_map_num
        self.first_index = sorted(self.mapping[self.map_num])[0]
        self.length = len(self.mapping[self.map_num])

    def __getitem__(self, index):
        if self.train:
            index = self.mapping[self.map_num][index + self.first_index]
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
            index = self.first_index + index
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


class ThreeChannelProstateImages(Dataset):

    def __init__(self, train, device, length, width, height, depth):
        self.train = train
        self.device = device
        self.normalize = torchvision.transforms.Normalize(mean=0, std=1)
        self.length = length
        self.width = width
        self.height = height
        self.depth = depth

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.train:
            path = "{}/{}".format(
                "/home/andrewg/PycharmProjects/assignments/resampled_cropped/train",
                "{}/{}_{}.nrrd"
            )
            try:
                t2_path = path.format("t2", index, 0)
                t2_image = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
                cancer_label = 0
            except:
                t2_path = path.format("t2", index, 1)
                t2_image = sitk.ReadImage(t2_path)
                cancer_label = 1

            adc_path = path.format("adc", index, cancer_label)
            adc_image = sitk.GetArrayFromImage(sitk.ReadImage(adc_path))

            bval_path = path.format("bval", index, cancer_label)
            bval_image = sitk.GetArrayFromImage(sitk.ReadImage(bval_path))

            image = np.array([t2_image, adc_image, bval_image])

            # The last digit in the file name specifies cancer/non-cancer
            output = {"image": image, "cancer": cancer_label}

        else:
            path = "{}/{}".format(
                "/home/andrewg/PycharmProjects/assignments/resampled_cropped/test",
                "{}/{}.nrrd"
            )
            t2_image = sitk.GetArrayFromImage(sitk.ReadImage(path.format("t2", index)))
            adc_image = sitk.GetArrayFromImage(sitk.ReadImage(path.format("adc", index)))
            bval_image = sitk.GetArrayFromImage(sitk.ReadImage(path.format("bval", index)))
            image = np.array([t2_image, adc_image, bval_image])
            output = {"image": image}

        if output["image"].shape == (self.depth, self.height, self.width):
            output["image"] = None
        else:
            output["image"] = torch.from_numpy(output["image"]).float().to(self.device)
            output["image"] = self.normalize(output["image"])

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
        # data = nn.BatchNorm2d(32).cuda()(data)
        data = nn.ReLU()(data)
        data = self.conv2(data)
        # data = nn.BatchNorm2d(32).cuda()(data)
        data = nn.ReLU()(data)
        data = self.max_pool1(data)
        data = self.conv3(data)
        data = nn.BatchNorm2d(64).cuda()(data)
        data = nn.ReLU()(data)
        data = self.conv4(data)
        # data = nn.BatchNorm2d(64).cuda()(data)
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


class CNNMultiChannel(nn.Module):
    def __init__(self):
        pass

    def forward(self, data):
        return data


def train_model(train_data, val_data, model, epochs, optimizer, loss_function, show=False):
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
    :param num_folds: How many folds were chosen to be
    :param show: Whether or not the user wants to see plots of loss, f1, and auc scores for the training and validation
                 sets
    :return: None
    """

    errors = []
    eval_errors = []
    f1_train = []
    auc_train = []
    f1_eval = []
    auc_eval = []
    num_training_batches = len(train_data)
    for epoch in range(epochs):
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
        train_loss_avg = train_loss / num_training_batches
        f1_train.append(f1_score(all_actual, all_preds))
        fpr, tpr, _ = roc_curve(all_actual, all_preds, pos_label=1)
        auc_train.append(auc(fpr, tpr))
        errors.append(train_loss_avg)
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
        eval_loss_avg = eval_loss / num_val_batches
        print("Loss Epoch {}, Training: {}, Validation: {}".format(epoch + 1, train_loss_avg, eval_loss_avg))
        f1_eval.append(f1_score(all_actual, all_preds))
        fpr, tpr, _ = roc_curve(all_actual, all_preds, pos_label=1)
        auc_eval.append(auc(fpr, tpr))
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
    return auc_train[-1], f1_train[-1], auc_eval[-1], f1_eval[-1]


def k_fold_cross_validation(K, train_data, val_data, epochs, loss_function, show=True):
    train_data, train_dataloader = train_data
    val_data, val_dataloader = val_data
    auc_train_avg, f1_train_avg, auc_eval_avg, f1_eval_avg = [], [], [], []
    models = []
    for k in range(0,1):
        print("Fold {}".format(k))
        model = CNN()
        model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.06)
        he_initialize(model)
        # optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.05)
        train_data.change_map_num(k)
        val_data.change_map_num(k)
        auc_train, f1_train, auc_eval, f1_eval = train_model(train_dataloader, val_dataloader, model, epochs, optimizer,
                                                             loss_function, show=True)
        auc_train_avg.append(auc_train)
        f1_train_avg.append(f1_train)
        auc_eval_avg.append(auc_eval)
        f1_eval_avg.append(f1_eval)
        models.append(model)

    scores = [auc_train_avg, f1_train_avg, auc_eval_avg, f1_eval_avg]
    if show:
        print(scores)
    return list(zip(models, scores))


def test_predictions(dataloader, model):
    """
    This function runs the model on the batches in the test set and returns a dataframe with ProxID, fid, and ClinSig
    columns. The predictions x <- ClinSig, 0 <= x <= 1, x <- R.
    :param dataloader: The data loader with the test batches
    :param model: The trained pytorch model
    :return: A dataframe as described above
    """

    predictions = pd.read_csv(r"/home/andrewg/PycharmProjects/assignments/ProstateX-TestLesionInformation/ProstateX-Findings-Test.csv")
    predictions.insert(4, "ClinSig", 0)
    predictions = predictions.drop(["pos", "zone"], axis=1)

    end_batch = 0
    for idx, batch in enumerate(dataloader):
        outputs = model(batch["image"])
        start_batch = end_batch
        end_batch = start_batch + len(outputs)
        predictions["ClinSig"].iloc[start_batch: end_batch] = outputs.flatten().tolist()
    return predictions


def he_initialize(model):
    if isinstance(model, nn.Conv2d):
        torch.nn.init.kaiming_normal_(model.weight)
        if model.bias:
            torch.nn.init.kaiming_normal_(model.bias)
    if isinstance(model, nn.Linear):
        torch.nn.init.kaiming_normal_(model.weight)
        if model.bias:
            torch.nn.init.kaiming_normal_(model.bias)


if __name__ == "__main__":
    # Define hyper-parameters
    batch_size_train = 200
    batch_size_val = 50
    batch_size_test = 50
    loss_function = nn.BCELoss().cuda()

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    modality = "bval"
    image_folder_contents = os.listdir("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/{}".format(
                                                                                                            modality))
    num_images = len(image_folder_contents)

    with open("/home/andrewg/PycharmProjects/assignments/train_key_mappings.pkl", "rb") as f:
        train_key_mappings = pk.load(f)
    with open("/home/andrewg/PycharmProjects/assignments/fold_key_mappings.pkl", "rb") as f:
        fold_key_mappings = pk.load(f)

    p_images_train = ProstateImages(modality=modality, train=True, device=device,
                                    mapping=train_key_mappings)

    p_images_validation = ProstateImages(modality=modality, train=True, device=device,
                                         mapping=fold_key_mappings)

    dataloader_train = DataLoader(p_images_train, batch_size=batch_size_train, shuffle=True)
    dataloader_val = DataLoader(p_images_validation, batch_size=batch_size_val)

    # Model 1
    cnn = CNN()
    cnn.cuda()

    models_and_scores = k_fold_cross_validation(K=5, train_data=(p_images_train, dataloader_train),
                                                val_data=(p_images_validation, dataloader_val), epochs=19,
                                                loss_function=loss_function, show=True)
    p_images_test = ProstateImages(modality=modality, train=False, device=device)
    dataloader_test = DataLoader(p_images_test, batch_size=batch_size_test, shuffle=False)

    results = test_predictions(dataloader_test, models_and_scores[0][0])
    torch.save(models_and_scores[0][0].state_dict(),
               "/home/andrewg/PycharmProjects/assignments/predictions/best_model.pt")

    results.to_csv("/home/andrewg/PycharmProjects/assignments/predictions/preds2.csv")
