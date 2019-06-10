import pandas as pd
import torch
from models import CNN
import os
import numpy as np
import SimpleITK as sitk
from sklearn.metrics import auc, roc_curve, f1_score

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    cuda_destination=1
    kgh_cropped_images_directory = "/home/andrewg/PycharmProjects/assignments/resampled_cropped/kgh"
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
    tensors = torch.zeros(len(targets), 3, 32, 32).float().cuda(cuda_destination)
    normalize = sitk.NormalizeImageFilter()
    bad_indices = []
    for idx, pcad_id in enumerate(ids):
        for roi in ROIs:
            image_name = "{}_{}.nrrd".format(pcad_id, roi)
            if image_name in image_dir:
                image = normalize.Execute(sitk.ReadImage("{}/{}".format(image_path, image_name)))
                image = torch.from_numpy(sitk.GetArrayFromImage(image)
                                         .astype(np.float64)).float().unsqueeze(0).cuda(cuda_destination)
                tensors[idx] = image
                break
            if roi == "b_tz":
                bad_indices.append(idx)
    cnn = CNN(cuda_destination=1)
    cnn.load_state_dict(torch.load("/home/andrewg/PycharmProjects/assignments/predictions/models/1.pt"))
    cnn.cuda(cuda_destination)
    predictions = cnn(tensors).squeeze(1)
    for idx in bad_indices:
        predictions = torch.cat((predictions[0:idx], predictions[idx + 1:]), 0)
        targets = torch.cat((targets[0:idx], targets[idx + 1:]), 0)
    targets, rounded_targets = targets.tolist(), targets.round().tolist()
    predictions, rounded_predictions = predictions.tolist(), predictions.round().tolist()
    fpr, tpr, _ = roc_curve(targets, predictions, pos_label=1)
    print("The AUC is {}".format(auc(fpr, tpr)))
    print("The F1 score is {}".format(f1_score(rounded_targets, rounded_predictions)))
    print("The accuracy is {}".format(sum(np.array(targets) == np.array(rounded_predictions)) / len(targets)))

