import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models import CNN, CNN2, MNIST_CNN
from data_helpers import train_model, KGHProstateImages, change_requires_grad, nrrd_to_tensor
from adabound import AdaBound
import matplotlib.pyplot as plt
import torchvision

if __name__ == "__main__":

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cuda_destination = 0
    ngpu = 1
    device = torch.device("cuda:{}".format(cuda_destination) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    model = CNN2(0)
    model.cuda(0)
    model.load_state_dict(torch.load("/home/andrewg/PycharmProjects/assignments/predictions/models/30.pt",
                                     map_location=device))

    im = nrrd_to_tensor("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/bval/40_1.nrrd").to(
        device).float()
    class_activation = model.class_activation_mapping(im.unsqueeze(0))
    CNN2.visualize(im, class_activation, code=0)
    CNN2.visualize(im, class_activation, code=1)
    print(class_activation[1])

    exit()
    # Prepare the data
    data = KGHProstateImages(device=device, modality="bval")

    num_train = int(0.8 * len(data))
    num_val = len(data) - num_train
    training_data, testing_data = torch.utils.data.random_split(data, (num_train, num_val))
    train_loader = DataLoader(training_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(testing_data, batch_size=10)

    # Hyper-parameters
    num_layers = 9
    lr = 0.00001
    final_lr = lr * 100
    weight_decay = 0.05
    num_epochs = 100
    loss_function = nn.BCELoss().cuda(cuda_destination)
    # loss_function = nn.CrossEntropyLoss().cuda(cuda_destination)

    for num_layers_to_freeze in range(8, num_layers):
        print("{} layers frozen".format(num_layers_to_freeze))
        model = CNN(cuda_destination)
        model.cuda(cuda_destination)
        model.load_state_dict(torch.load("/home/andrewg/PycharmProjects/assignments/predictions/models/1.pt",
                                         map_location=device))
        for child_num in range(-1, -1 * (num_layers - num_layers_to_freeze + 1), -1):
            print("Reinitialized layer {}".format(num_layers + child_num + 1))
            nn.init.kaiming_normal_(list(model.children())[child_num].weight)
        change_requires_grad(model, num_layers_to_freeze, False)

        optimizer = AdaBound(model.parameters(), lr=lr, final_lr=final_lr, weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        
        conf_matrix, _, _, _, _ = train_model(train_loader, test_loader, model, num_epochs, optimizer,
                                              loss_function, softmax=False, show=True)
        print(conf_matrix)
