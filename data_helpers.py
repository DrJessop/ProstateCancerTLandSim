import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import SimpleITK as sitk
import numpy as np
import adabound
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, auc, roc_curve
import random
from image_augmentation import rotation3d
import shutil
import pandas as pd
import pickle as pk


def resample_image(itk_image, out_spacing, is_label=False):
    """
    Retrieved this function from:
    https://www.programcreek.com/python/example/96383/SimpleITK.sitkNearestNeighbor
    :param itk_image: The image that we would like to resample
    :param out_spacing: The new spacing of the voxels we would like
    :param is_label: If True, use kNearestNeighbour interpolation, else use BSpline
    :return: The re-sampled image
    """

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0]*(original_spacing[0]/out_spacing[0]))),
                int(np.round(original_size[1]*(original_spacing[1]/out_spacing[1]))),
                int(np.round(original_size[2]*(original_spacing[2]/out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkCosineWindowedSinc)

    return resample.Execute(itk_image)


def resample_all_images(modality, out_spacing, some_missing=False):
    """
    This function returns a list of re-sampled images for a given modality and desired spacing
    :param modality: ex. t2, adc, bval, etc.
    :param out_spacing: The desired spacing of the images
    :param some_missing: If an image may be missing, this may be set to True to handle the case
    of a missing image
    :return: Re-sampled images
    """

    if some_missing:
        return [resample_image(mod_image, out_spacing) if mod_image != "" else ""
                for mod_image in modality]
    return [resample_image(mod_image, out_spacing)
            if mod_image != "" else "" for mod_image in modality]


def crop_from_center(images, ijk_coordinates, width, height, depth, i_offset=0, j_offset=0):
    """
    Helper function for image cropper and rotated crop that produces a crop of dimension width x height x depth,
    where the lesion is offset by i_offset (x dimension) and j_offset (y dimension)
    :param images: A list of size 3 tuples, where the elements in the tuple are t2, adc, and bval SITK images
                   respectively
    :param ijk_coordinates: The coordinates of the lesion
    :param width: Desired width of the crop
    :param height: Desired height of the crop
    :param depth: Desired depth of the crop
    :param i_offset: Desired offset in pixels away from the lesion in the x direction
    :param j_offset: Desired offset in pixels away from the lesion in the y direction
    :return: The newly created crop
    """
    crop = [image[(ijk_coordinates[idx][0] - i_offset) - width // 2: (ijk_coordinates[idx][0] - i_offset)
                  + int(np.ceil(width / 2)),
                  (ijk_coordinates[idx][1] - j_offset) - height // 2: (ijk_coordinates[idx][1] - j_offset)
                  + int(np.ceil(height / 2)),
                  ijk_coordinates[idx][2] - depth // 2: ijk_coordinates[idx][2]
                  + int(np.ceil(depth / 2))]
            for idx, image in enumerate(images)]
    return crop


def rotated_crop(patient_images, crop_width, crop_height, crop_depth, degrees, lps, ijk_values, show_result=False):
    """
    This is a helper function for image_cropper, and it returns a crop around a rotated image
    :param patient_image: The sitk image that is to be cropped
    :param crop_width: The desired width of the crop
    :param crop_height: The desired height of the crop
    :param crop_depth: The desired depth of the crop
    :param degrees: A list of all allowable degrees of rotation (gets converted to radians in the rotation3d function
                    which is called below)
    :param lps: The region of interest which will be the center of rotation
    :param ijk_values: A list of lists, where each list is the ijk values for each image's biopsy position
    :param show_result: Whether or not the user wants to see the first slice of the new results
    :return: The crop of the rotated image
    """

    degree = np.random.choice(degrees)
    rotated_patient_images = list(map(lambda patient: rotation3d(patient, degree, lps), patient_images))

    i_offset = np.random.randint(-7, 7)
    j_offset = np.random.randint(-7, 7)

    crop = crop_from_center(rotated_patient_images, ijk_values, crop_width, crop_height, crop_depth, i_offset=i_offset,
                            j_offset=j_offset)

    if show_result:
        for i in range(3):
            plt.imshow(sitk.GetArrayFromImage(rotated_patient_images[0])[0], cmap="gray")
            plt.imshow(sitk.GetArrayFromImage(crop[i])[0], cmap="gray")
            plt.show()
        input()
    return crop


def write_cropped_images_train_and_folds(cropped_images, num_crops, num_folds=5, fold_fraction=0.2):
    """
    This function writes all cropped images to a training directory (for each modality) and creates a list of hashmaps
    for folds. These maps ensure that there is a balanced distribution of cancer and non-cancer in each validation set
    as well as the training set used for prediction.
    :param cropped_images: A dictionary where the keys are the patient IDs, and the values are lists where each element
    is a list of length three (first element in that list is t2 image, and then adc and bval).
    :param num_crops: The number of crops for a given patient's image
    :param num_folds: The number of sets to be created
    :param fold_fraction: The amount of cancer patients to be within a fold's validation set
    :return: fold key and train key mappings (lists of hash functions which map to the correct patient data)
    """

    destination = r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/"

    directory_contents = os.listdir(destination)
    for sub_directory in directory_contents:
        sub_directory_path = destination + sub_directory
        shutil.rmtree(sub_directory_path)
        os.mkdir(sub_directory_path)

    destination = destination + r"{}/{}_{}.nrrd"

    patient_images = [(key, patient_image) for key in cropped_images.keys()
                      for patient_image in cropped_images[key]]

    for p_id in range(len(patient_images)):
        _, (patient_image, cancer_marker) = patient_images[p_id]
        sitk.WriteImage(patient_image[0], destination.format("t2", p_id, cancer_marker))
        sitk.WriteImage(patient_image[1], destination.format("adc", p_id, cancer_marker))
        sitk.WriteImage(patient_image[2], destination.format("bval", p_id, cancer_marker))

    patient_indices = set(range(len(patient_images) // num_crops))
    non_cancer_patients = {idx for idx in patient_indices if patient_images[idx * num_crops][0][-1] == '0'}
    cancer_patients = {idx for idx in patient_indices if patient_images[idx * num_crops][0][-1] == '1'}

    num_each_class_fold = int(fold_fraction * len(cancer_patients))

    fold_key_mappings = []
    train_key_mappings = []
    for k in range(num_folds):

        non_cancer_in_fold = random.sample(non_cancer_patients, num_each_class_fold)
        cancer_in_fold = random.sample(cancer_patients, num_each_class_fold)

        fold_set = set()
        fold_set.update(non_cancer_in_fold)
        fold_set.update(cancer_in_fold)

        out_of_fold = patient_indices.difference(fold_set)

        # Uses up all the cancer patients
        cancer_out_of_fold = {idx for idx in out_of_fold if patient_images[idx * num_crops][0][-1] == '1'}
        non_cancer_out_of_fold = random.sample(out_of_fold.difference(cancer_out_of_fold), len(cancer_out_of_fold))

        out_of_fold_set = set()
        out_of_fold_set.update(cancer_out_of_fold)
        out_of_fold_set.update(non_cancer_out_of_fold)

        # Prepare fold indices
        fold_image_indices = set()
        for key in fold_set:
            image_index = key * num_crops
            for pos in range(num_crops):
                fold_image_indices.add(image_index + pos)

        # Prepare train key indices
        out_of_fold_image_indices = set()
        for key in out_of_fold_set:
            image_index = key * num_crops
            for pos in range(num_crops):
                out_of_fold_image_indices.add(image_index + pos)

        fold_key_mapping = {}
        key = 0
        for fold_image_index in fold_image_indices:
            fold_key_mapping[key] = fold_image_index
            key += 1

        train_key_mapping = {}
        key = 0
        for train_image_index in out_of_fold_image_indices:
            train_key_mapping[key] = train_image_index
            key += 1

        fold_key_mappings.append(fold_key_mapping)
        train_key_mappings.append(train_key_mapping)

    return fold_key_mappings, train_key_mappings


def image_cropper(findings_dataframe, resampled_images, padding,
                  crop_width, crop_height, crop_depth, num_crops_per_image=1, train=True):
    """
    Given a dataframe with the findings of cancer, a list of images, and a desired width, height,
    and depth, this function returns a set of cropped versions of the original images of dimension
    crop_width x crop_height x crop_depth
    :param findings_dataframe: A pandas dataframe containing the LPS coordinates of the cancer
    :param resampled_images: A list of images that have been resampled to all have the same
                             spacing
    :param padding: 0-Padding in the i,j,k directions
    :param crop_width: The desired width of a patch
    :param crop_height: The desired height of a patch
    :param crop_depth: The desired depth of a patch
    :param num_crops_per_image: The number of crops desired for a given image
    :param train: Boolean, represents whether these are crops of the training or the test set
    :return: A list of cropped versions of the original re-sampled images
    """

    t2_resampled, adc_resampled, bval_resampled = resampled_images

    if num_crops_per_image < 1:
        print("Cannot have less than 1 crop for an image")
        exit()
    degrees = [5, 10, 15, 20, 25, 180]  # One of these is randomly chosen for every rotated crop
    crops = {}
    invalid_keys = set()
    for _, patient in findings_dataframe.iterrows():
        patient_id = patient["patient_id"]
        patient_images = [t2_resampled[int(patient_id[-4:])], adc_resampled[int(patient_id[-4:])],
                          bval_resampled[int(patient_id[-4:])]]
        if train:
            cancer_marker = int(patient["ClinSig"])  # 1 if cancer, else 0
        if '' in patient_images:  # One of the images is blank
            continue
        else:
            # Adds padding to each of the images
            patient_images = [padding.Execute(p_image) for p_image in patient_images]
            lps = [float(loc) for loc in patient["pos"].split(' ') if loc != '']

            # Convert lps to ijk for each of the images
            ijk_vals = [patient_images[idx].TransformPhysicalPointToIndex(lps) for idx in range(3)]

            # Below code makes a crop of dimensions crop_width x crop_height x crop_depth
            for crop_num in range(num_crops_per_image):
                if crop_num == 0:  # The first crop we want to guarantee has the biopsy position exactly in the center
                    crop = crop_from_center(patient_images, ijk_vals, crop_width, crop_height, crop_depth)
                else:
                    # Rotate the image, and then translate and crop
                    crop = rotated_crop(patient_images, crop_width, crop_height, crop_depth, degrees, lps, ijk_vals)
                invalid_sizes = [im.GetSize() for im in crop if im.GetSize() != (crop_width, crop_height, crop_depth)]
                if train:
                    if invalid_sizes:  # If not all of the image sizes are correct
                        print("Invalid image for patient {}".format(patient_id))
                        invalid_keys.add("{}_{}".format(patient_id, cancer_marker))
                        continue
                    # If any of the crops are bad, they're all bad
                    elif np.sum(sitk.GetArrayFromImage(crop[0]).flatten()) == 0:
                        invalid_keys.add("{}_{}".format(patient_id, cancer_marker))
                else:
                    print(np.sum(sitk.GetArrayFromImage(crop[2]).flatten()))
                    if np.sum(sitk.GetArrayFromImage(crop[0]).flatten()) == 0:
                        crop = [sitk.GetImageFromArray(np.random.rand(crop_depth, crop_height, crop_width))
                                for _ in range(3)]
                        print(patient_id)
                if train:
                    key = "{}_{}".format(patient_id, cancer_marker)
                else:
                    key = patient_id
                if key in crops.keys():
                    if train:
                        crops[key].append((crop, cancer_marker))
                    else:
                        crops[key].append(crop)
                else:
                    if train:
                        crops[key] = [(crop, cancer_marker)]
                    else:
                        crops[key] = [crop]

    for key in invalid_keys:
        crops.pop(key)
    return crops


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
        if np.isnan(output["image"][0, 0, 0]):
            output["image"] = np.random.rand(3, 32, 32)
        output["image"] = torch.from_numpy(output["image"]).float().to(self.device)
        output["index"] = index
        return output


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
            images, class_vector = batch["image"], batch["cancer"].float().cuda(model.cuda_destination).unsqueeze(1)
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
                images, class_vector = batch["image"], batch["cancer"].float().cuda(model.cuda_destination).unsqueeze(1)
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
                            weight_decay=0.04, show=True, cuda_destination=1):
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
    :param cuda_destination: The GPU that is used by the model
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
        model = network(cuda_destination)
        model.cuda(model.cuda_destination)
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


def prepare_kgh_data(cuda_destination, device):
    kgh_labels_file = "/home/andrewg/PycharmProjects/assignments/data/KGHData/kgh.csv"
    cancer_labels = pd.read_csv(kgh_labels_file)[["anonymized", "Total Gleason Xypeguide"]]
    cancer_labels = cancer_labels.drop([7, 9, 14, 18, 35])
    for idx, val in cancer_labels["Total Gleason Xypeguide"].iteritems():
        if val == '0':
            cancer_labels["Total Gleason Xypeguide"][idx] = 0
        elif str(val) in '123456789':
            cancer_labels["Total Gleason Xypeguide"][idx] = 1
    valid = set(cancer_labels[cancer_labels["Total Gleason Xypeguide"] == 0].index)
    valid.update(cancer_labels[cancer_labels["Total Gleason Xypeguide"] == 1].index)
    cancer_labels = cancer_labels.loc[valid]
    image_path = "/home/andrewg/PycharmProjects/assignments/resampled_cropped/kgh"
    image_dir = set(os.listdir(image_path))
    ROIs = ["tz", "pz", "tx", "b_tz"]

    ids = cancer_labels["anonymized"]
    targets = torch.from_numpy(np.asarray(cancer_labels["Total Gleason Xypeguide"], np.float64))
    tensor = torch.zeros(len(targets), 3, 32, 32)
    normalize = sitk.NormalizeImageFilter()
    bad_indices = []
    for idx, pcad_id in enumerate(ids):
        for roi in ROIs:
            image_name = "{}_{}.nrrd".format(pcad_id, roi)
            if image_name in image_dir:
                image = normalize.Execute(sitk.ReadImage("{}/{}".format(image_path, image_name)))
                image = torch.from_numpy(sitk.GetArrayFromImage(image)).float().to(device)
                tensor[idx] = image
                break
            if roi == "b_tz":
                bad_indices.append(idx)
    tensor = tensor.float().cuda(cuda_destination)
    torch.save(tensor, "/home/andrewg/PycharmProjects/assignments/kgh_data_tensor.pt")
    torch.save(targets, "/home/andrewg/PycharmProjects/assignments/kgh_target_tensor.pt")
    with open("/home/andrewg/PycharmProjects/assignments/bad_indices.pkl", "wb") as f:
        pk.dump(bad_indices, f)
