import pandas as pd
import torch
import torch.utils.data
from models import CNN
import os
from sklearn.metrics import auc, roc_curve
from data_helpers import prepare_kgh_data
import pickle as pk


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    cuda_destination = 0
    ngpu = 1
    device = torch.device("cuda:{}".format(cuda_destination) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    model = CNN(cuda_destination=cuda_destination)
    model.load_state_dict(torch.load("/home/andrewg/PycharmProjects/assignments/predictions/models/1.pt",
                                     map_location=device))

    tensor = "kgh_data_tensor.pt"
    targets = "kgh_target_tensor.pt"
    bad_indices = "bad_indices.pkl"
    if not({tensor, targets, bad_indices}.issubset(os.listdir("/home/andrewg/PycharmProjects/assignments"))):
        prepare_kgh_data(cuda_destination, device)
    tensor = torch.load("/home/andrewg/PycharmProjects/assignments/kgh_data_tensor.pt")
    targets = torch.load("/home/andrewg/PycharmProjects/assignments/kgh_target_tensor.pt")
    with open("/home/andrewg/PycharmProjects/assignments/bad_indices.pkl", "rb") as f:
        bad_indices = pk.load(f)
    model.cuda(cuda_destination)
    predictions = model(tensor)
    predictions = [prediction.cpu().detach().numpy() for idx, prediction in enumerate(predictions) if idx not in
                   bad_indices]
    targets = [target.cpu().detach().numpy() for idx, target in enumerate(targets) if idx not in bad_indices]
    fpr, tpr, _ = roc_curve(targets, predictions, pos_label=1)
    print("The AUC is {}".format(auc(fpr, tpr)))

