import os
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, auc, roc_curve
import pandas as pd
import pickle as pk
import numpy as np
import adabound


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
    global mean_tensor
    global standard_deviation_tensor

    def __init__(self, modality, train, device, mapping=None):
        self.modality = modality
        self.train = train
        self.device = device
        self.normalize = sitk.NormalizeImageFilter()
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
        # output["image"] = (output["image"] - mean_tensor) / standard_deviation_tensor
        if np.isnan(output["image"][0, 0, 0]):
            output["image"] = np.random.rand(3, 32, 32)
        output["image"] = torch.from_numpy(output["image"]).float().to(self.device)
        output["index"] = index
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
        # data = nn.BatchNorm2d(32).cuda(cuda_destination)(data)
        data = nn.ReLU()(data)
        data = self.conv2(data)
        data = nn.BatchNorm2d(32).cuda(cuda_destination)(data)
        data = nn.ReLU()(data)
        data = self.max_pool1(data)
        data = self.conv3(data)
        data = nn.BatchNorm2d(64).cuda(cuda_destination)(data)
        data = nn.ReLU()(data)
        data = self.conv4(data)
        data = nn.ReLU()(data)
        # data = nn.BatchNorm2d(64).cuda(cuda_destination)(data)
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
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(2, 2, 2), stride=2, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(2, 2), stride=1)
        self.linear1 = nn.Linear(in_features=288, out_features=100)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(in_features=100, out_features=1)

    def forward(self, data):
        # print(data.shape)
        data = data.unsqueeze(1)
        # print(data.shape)
        data = self.conv1(data)
        data = nn.ELU()(data)
        data = nn.ELU()(data)
        data = nn.BatchNorm3d(32).cuda(cuda_destination)(data)
        # print(data.shape)
        data = self.pool1(data)
        # print(data.shape)
        data = data.squeeze(2)
        # print(data.shape)
        data = self.conv2(data)
        data = nn.ELU()(data)
        data = nn.ELU()(data)
        # data = nn.BatchNorm2d(16).cuda(cuda_destination)(data)
        # print(data.shape)
        data = self.conv3(data)
        data = nn.ELU()(data)
        data = nn.ELU()(data)
        # data = nn.BatchNorm2d(8).cuda(cuda_destination)(data)
        # print(data.shape)
        data = data.view(-1, 8 * 6 * 6)
        # print(data.shape)
        data = self.linear1(data)
        data = nn.ELU()(data)
        data = nn.ELU()(data)
        # data = nn.BatchNorm1d(100).cuda(cuda_destination)(data)
        # print(data.shape)
        data = self.dropout(data)
        data = self.linear2(data)
        # print(data.shape)
        data = nn.Sigmoid()(data)
        # print(data.shape)
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
            images, class_vector = batch["image"], batch["cancer"].float().cuda(cuda_destination).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(images)
            if sum(torch.isnan(preds)):
                print(batch["index"])
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
                images, class_vector = batch["image"], batch["cancer"].float().cuda(cuda_destination).unsqueeze(1)
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


def k_fold_cross_validation(network, K, train_data, val_data, epochs, loss_function, lr=0.005, momentum=0.9,
                            weight_decay=0.04, show=True):
    """
    Given training and validation data, performs K-fold cross-validation.
    :param network: Instance of the class you will use as the network
    :param K: Number of folds
    :param train_data: A tuple containing a ProstateImages object where train=True and a dataloader in which
                       the ProstateImages object is supplied as a parameter
    :param val_data: A tuple containing a ProstateImages object where train=True and a dataloader in which
                     the ProstateImages object is supplied as a parameter
    :param epochs: The number of epochs each model is to be trained for
    :param loss_function: The desired loss function which is to be used by every model being trained
    :param lr: The learning rate, default is 0.005
    :param momentum: The momentum for stochastic gradient descent, default is 0.9
    :param weight_decay: L2 regularization alpha parameter, default is 0.06
    :param show: Whether or not to show the train/val loss, f1, and auc curves after each fold, default is True
    :return: A list (size 4) of lists, where the first list contains the auc scores for the training sets, the second
             list contains the f1 scores for the training sets, the third list contains the auc scores for the
             validation sets, and the fourth and final list contains the f1 scores for the validation sets
    """
    train_data, train_dataloader = train_data
    val_data, val_dataloader = val_data
    auc_train_avg, f1_train_avg, auc_eval_avg, f1_eval_avg = [], [], [], []
    models = []
    for k in range(1):
        print("Fold {}".format(k + 1))
        model = network()
        model.cuda(cuda_destination)
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = adabound.AdaBound(model.parameters(), lr=lr, final_lr=lr*100, weight_decay=weight_decay)
        he_initialize(model)
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

    with open("/home/andrewg/PycharmProjects/assignments/predictions/k_fold_statistics/stats.txt", "a") as f:
        f.write("AUC train average\n")
        for avg in auc_train_avg:
            f.write(str(avg) + '\n')
        f.write("F1 train average\n")
        for avg in f1_train_avg:
            f.write(str(avg) + '\n')
        f.write("AUC eval average\n")
        for avg in auc_eval_avg:
            f.write(str(avg) + '\n')
        f.write("F1 eval average\n")
        for avg in f1_eval_avg:
            f.write(str(avg) + '\n')
    return list(zip(models, scores))


def test_predictions(dataloader, model):
    """
    This function runs the model on the batches in the test set and returns a dataframe with ProxID, fid, and ClinSig
    columns. The predictions x <- ClinSig, 0 <= x <= 1, x <- R.
    :param dataloader: The data loader with the test batches
    :param model: The trained pytorch model
    :return: A dataframe as described above
    """

    model.eval()
    predictions = pd.read_csv(r"/home/andrewg/PycharmProjects/assignments/ProstateX-TestLesionInformation/ProstateX-Findings-Test.csv")
    predictions.insert(4, "ClinSig", 0)
    predictions = predictions.drop(["pos", "zone"], axis=1)
    end_batch = 0

    for idx, batch in enumerate(dataloader):
        outputs = model(batch["image"])
        print(outputs)
        start_batch = end_batch
        end_batch = start_batch + len(outputs)
        predictions["ClinSig"].iloc[start_batch: end_batch] = outputs.flatten().tolist()
    return predictions


def he_initialize(model):
    """
    He weight initialization, as described in Delving Deep into Rectifiers:Surpassing Human-Level Performance on
    ImageNet Classification (https://arxiv.org/pdf/1502.01852.pdf)
    :param model: The network being initialized
    :return: None
    """
    if isinstance(model, nn.Conv2d):
        torch.nn.init.kaiming_normal_(model.weight)
        if model.bias:
            torch.nn.init.kaiming_normal_(model.bias)
    if isinstance(model, nn.Linear):
        torch.nn.init.kaiming_normal_(model.weight)
        if model.bias:
            torch.nn.init.kaiming_normal_(model.bias)


'''
cuda_destination = 1
testing_model = CNN()
testing_model.cuda(1)
device = torch.device("cuda:{}".format(1) if (torch.cuda.is_available()) else "cpu")
testing_model.load_state_dict(torch.load("/home/andrewg/PycharmProjects/assignments/predictions/models/1.pt",
                                         map_location=device))
ngpu = 1
test_set = ProstateImages(modality="bval", train=False, device=device)
loader = DataLoader(test_set, batch_size=5, shuffle=False)
batch = next(iter(loader))
print(testing_model(batch["image"]))
'''

if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Define hyper-parameters
    cuda_destination = 1
    batch_size_train = 100
    batch_size_val = 50
    batch_size_test = 50
    loss_function = nn.BCELoss().cuda(cuda_destination)

    ngpu = 1
    device = torch.device("cuda:{}".format(cuda_destination) if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    modality = "bval"
    image_folder_contents = os.listdir("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/{}".format(
                                                                                                            modality))

    with open("/home/andrewg/PycharmProjects/assignments/train_key_mappings.pkl", "rb") as f:
        train_key_mappings = pk.load(f)
    with open("/home/andrewg/PycharmProjects/assignments/fold_key_mappings.pkl", "rb") as f:
        fold_key_mappings = pk.load(f)

    f = "/home/andrewg/PycharmProjects/assignments/resampled_cropped/train"
    with open("{}/{}".format(f, "{}_mean_tensor.npy".format(modality)), "rb") as g:
        mean_tensor = np.load(g)
    with open("{}/{}".format(f, "{}_std_tensor.npy".format(modality)), "rb") as h:
        standard_deviation_tensor = np.load(h)

    p_images_train = ProstateImages(modality=modality, train=True, device=device,
                                    mapping=train_key_mappings)

    p_images_validation = ProstateImages(modality=modality, train=True, device=device,
                                         mapping=fold_key_mappings)

    dataloader_train = DataLoader(p_images_train, batch_size=batch_size_train, shuffle=True)
    dataloader_val = DataLoader(p_images_validation, batch_size=batch_size_val)

    # models_and_scores = k_fold_cross_validation(CNN2, K=5, train_data=(p_images_train, dataloader_train),
    #                                             val_data=(p_images_validation, dataloader_val), epochs=30,
    #                                             loss_function=loss_function, lr=0.001, show=True,
    #                                             weight_decay=0.05)
    p_images_test = ProstateImages(modality=modality, train=False, device=device)
    dataloader_test = DataLoader(p_images_test, batch_size=batch_size_test, shuffle=False)

    model = CNN()
    model.load_state_dict(torch.load("/home/andrewg/PycharmProjects/assignments/predictions/models/1.pt",
                                      map_location=device))
    model.cuda(cuda_destination)

    results = test_predictions(dataloader_test, model)

    model_dir = "/home/andrewg/PycharmProjects/assignments/predictions/models"
    predictions_dir = "/home/andrewg/PycharmProjects/assignments/predictions/prediction_files"
    sort_key = lambda file_name: int(file_name.split('.')[0])
    model_files = [f for f in os.listdir(model_dir) if f[0] in '123456789']
    results_files = [f for f in os.listdir(predictions_dir) if f[0] in '123456789']
    model_files = sorted(model_files, key=sort_key)
    results_files = sorted(results_files, key=sort_key)

    if model_files:
        next_model, _ = model_files[-1].split('.')
        next_model = "{}.pt".format(int(next_model) + 1)
    else:
        next_model = "1.pt"

    if results_files:
        next_result, _ = results_files[-1].split('.')
        next_result = "{}.csv".format(int(next_result) + 1)
    else:
        next_result = "1.csv"

    # torch.save(models_and_scores[0][0].state_dict(), "{}/{}".format(model_dir, next_model))
    results.to_csv("{}/{}".format(predictions_dir, next_result))
