import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models import CNN
from data_helpers import train_model, KGHProstateImages
from adabound import AdaBound


if __name__ == "__main__":

    cuda_destination = 0
    model = CNN(cuda_destination)
    model.cuda(cuda_destination)
    ngpu = 1
    device = torch.device("cuda:{}".format(cuda_destination) if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # model.load_state_dict(torch.load("/home/andrewg/PycharmProjects/assignments/predictions/models/1.pt",
    #                                  map_location=device))
    num_epochs = 40

    data = KGHProstateImages(device=device)
    num_train = int(0.2 * len(data))
    num_val = len(data) - num_train
    training_data, testing_data = torch.utils.data.random_split(data, (num_train, num_val))

    train_loader = DataLoader(training_data, batch_size=10)
    test_loader = DataLoader(testing_data, batch_size=20)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.05)
    lr = 0.00001
    optimizer = AdaBound(model.parameters(), lr=lr, final_lr=lr*1000, weight_decay=0.05)
    loss_function = nn.BCELoss().cuda(cuda_destination)
    train_model(train_loader, test_loader, model, num_epochs, optimizer, loss_function, show=True)

    # torch.save(model.state_dict(),
    #            "/home/andrewg/PycharmProjects/assignments/predictions/retrained_models/retrained.pt")