import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models import CNN
import random
from data_helpers import train_model


class KGHProstateImages(Dataset):
    def __init__(self, dictionary):
        self.dict = dictionary

    def __len__(self):
        return len(self.dict["image"])

    def __getitem__(self, idx):
        image_dict, cancer_dict = self.dict["image"], self.dict["cancer"]
        return {"image": image_dict[idx], "cancer": cancer_dict[idx]}


if __name__ == "__main__":
    cuda_destination = 0
    model = CNN(cuda_destination)
    model.cuda(cuda_destination)
    ngpu = 1
    device = torch.device("cuda:{}".format(cuda_destination) if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    model.load_state_dict(torch.load("/home/andrewg/PycharmProjects/assignments/predictions/models/1.pt",
                                     map_location=device))
    tensor = torch.load("/home/andrewg/PycharmProjects/assignments/kgh_data_tensor.pt")
    targets = torch.load("/home/andrewg/PycharmProjects/assignments/kgh_target_tensor.pt")
    num_cohorts = len(targets)
    training_indices = random.sample(range(num_cohorts), int(0.3 * num_cohorts))
    testing_indices = list(set(range(num_cohorts)).difference(training_indices))

    train_dict = {"image": {}, "cancer": {}}

    train_dict_position = 0
    for training_index in training_indices:
        train_dict["image"][train_dict_position] = tensor[training_index]
        train_dict["cancer"][train_dict_position] = targets[training_index]
        train_dict_position += 1

    test_dict = {"image": {}, "cancer": {}}

    test_dict_position = 0
    for testing_index in testing_indices:
        test_dict["image"][test_dict_position] = tensor[testing_index]
        test_dict["cancer"][test_dict_position] = targets[testing_index]
        test_dict_position += 1

    num_epochs = 150

    training_data = KGHProstateImages(train_dict)
    testing_data = KGHProstateImages(test_dict)

    train_loader = DataLoader(training_data, batch_size=10)
    test_loader = DataLoader(testing_data, batch_size=20)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.000001, momentum=0.9, weight_decay=0.02)
    loss_function = nn.BCELoss().cuda(cuda_destination)
    train_model(train_loader, test_loader, model, num_epochs, optimizer, loss_function, show=True)

    torch.save(model.state_dict(),
               "/home/andrewg/PycharmProjects/assignments/predictions/retrained_models/retrained.pt")
