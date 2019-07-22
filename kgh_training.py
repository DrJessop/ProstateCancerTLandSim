import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models import CNN, CNN2
from data_helpers import train_model, KGHProstateImages, change_requires_grad, flatten_batch
from adabound import AdaBound
import random


def train_test_split(data, num_crops_per_image, percent_cancer=0.5, percent_non_cancer=0.5):
    cancer_indices = list()
    non_cancer_indices = list()

    for i in range(len(data)):
        if data[i]["cancer"] == 0:
            non_cancer_indices.append(i)
        else:
            cancer_indices.append(i)

    rand_indices_cancer = len(cancer_indices) // num_crops_per_image
    random_patients_cancer = random.sample(range(rand_indices_cancer), int(percent_cancer * rand_indices_cancer))
    cancer_patients_train = [cancer_indices[(idx * num_crops_per_image): (idx * num_crops_per_image)
                             + num_crops_per_image] for idx in random_patients_cancer]
    cancer_patients_train = [idx for patient in cancer_patients_train for idx in patient]

    rand_indices_non_cancer = len(non_cancer_indices) // num_crops_per_image
    random_patients_non_cancer = random.sample(range(rand_indices_non_cancer), int(percent_non_cancer * rand_indices_non_cancer))
    non_cancer_patients_train = [non_cancer_indices[(idx * num_crops_per_image): (idx * num_crops_per_image)
                                 + num_crops_per_image] for idx in random_patients_non_cancer]
    non_cancer_patients_train = [idx for patient in non_cancer_patients_train for idx in patient]

    train_indices = cancer_patients_train + non_cancer_patients_train
    test_indices = set(non_cancer_indices + cancer_indices).difference(non_cancer_patients_train
                                                                       + cancer_patients_train)

    training_data = torch.utils.data.Subset(data, list(train_indices))
    testing_data = torch.utils.data.Subset(data, list(test_indices))
    return training_data, testing_data


def kgh_experiment(cnn_type, best_model, loss_function, hyperparameters, starting_layer=8, ending_layer=None,
                   optimizer="adabound", re_init=True):

    assert optimizer in ["adabound", "sgd"]

    if isinstance(cnn_type, CNN):
        model_type = "CNN"
    else:
        model_type = "CNN2"

    num_layers, lr, final_lr, weight_decay, num_epochs = hyperparameters

    for num_layers_to_freeze in range(starting_layer, ending_layer):
        print("{} layers frozen".format(num_layers_to_freeze))
        model = cnn_type(cuda_destination)
        model.cuda(cuda_destination)
        model.load_state_dict(torch.load(
            "/home/andrewg/PycharmProjects/assignments/predictions/models/{}/{}/{}.pt" .format("bval", "CNN",
                                                                                               1),
            map_location=device))

        if re_init:
            for child_num in range(-1, -1 * (num_layers - num_layers_to_freeze + 1), -1):
                print("Reinitialized layer {}".format(num_layers + child_num + 1))
                nn.init.kaiming_normal_(list(model.children())[child_num].weight)

            change_requires_grad(model, num_layers_to_freeze, False)
        if optimizer == "adabound":
            optimizer = AdaBound(model.parameters(), lr=lr, final_lr=final_lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        model, conf_matrix, _, _, _, _ = train_model(train_loader, test_loader, model, num_epochs, optimizer,
                                                     loss_function, softmax=softmax, show=True)
        print(conf_matrix)

    if isinstance(model, CNN2):
        num_images = 5
        images, class_vector = list(zip(*[(testing_data[i]["image"], testing_data[i]["cancer"])
                                          for i in range(num_images)]))
        images = torch.stack(images)
        class_vector = torch.tensor(class_vector)
        images, class_vector = flatten_batch(images.shape, images, class_vector, cuda_destination=0)
        for i in range(num_images):
            activation = model.class_activation_mapping(images[i * 20].unsqueeze(0))
            CNN2.visualize(images[i * 20], activation)
            print(activation[1])
        print([label for idx, label in enumerate(class_vector) if idx % 30 == 0])


if __name__ == "__main__":

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cuda_destination = 0
    ngpu = 1
    device = torch.device("cuda:{}".format(cuda_destination) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Prepare the data
    num_crops_per_image = 20
    modality = "adc"
    data = KGHProstateImages(device=device, modality=modality)

    # training_data, testing_data = train_test_split(data, 20, percent_cancer=0.8, percent_non_cancer=0.8)
    num_train = int(0.5 * len(data))
    num_val = len(data) - num_train
    training_data, testing_data = torch.utils.data.random_split(data, (num_train, num_val))
    train_loader = DataLoader(training_data, batch_size=5, shuffle=True)
    test_loader = DataLoader(testing_data, batch_size=5)

    softmax = False

    if softmax:
        cnn_type = CNN2
        best_model = 46
        loss_function = nn.CrossEntropyLoss().cuda(cuda_destination)
    else:
        cnn_type = CNN
        best_model = 1
        loss_function = nn.BCELoss().cuda(cuda_destination)

    num_layers = 9
    num_epochs = 100

    possible_lr = (0.000001 + 0.0000005 * i for i in range(3))
    for lr in possible_lr:
        final_lr = 100 * lr
        possible_weight_decay = (0.0001 + 0.00005 * i for i in range(3))
        for weight_decay in possible_weight_decay:
            hyperparameters = [
                num_layers,
                lr,
                final_lr,
                weight_decay,
                num_epochs
            ]

            print("Learning rate: {}, Weight decay: {}".format(lr, weight_decay))
            kgh_experiment(cnn_type, best_model, loss_function, hyperparameters, starting_layer=8,
                           ending_layer=9, re_init=False)

