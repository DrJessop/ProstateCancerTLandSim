import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models import CNN, CNN2
from data_helpers import train_model, KGHProstateImages, change_requires_grad, flatten_batch, he_initialize
from adabound import AdaBound
import random


if __name__ == "__main__":

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cuda_destination = 0
    ngpu = 1
    device = torch.device("cuda:{}".format(cuda_destination) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Prepare the data
    data = KGHProstateImages(device=device, modality="bval")

    '''
    cancer_indices = set()
    non_cancer_indices = set()

    for i in range(len(data)):
        if data[i]["cancer"] == 0:
            non_cancer_indices.add(i)
        else:
            cancer_indices.add(i)

    train_indices = set(random.sample(non_cancer_indices, int(0.5 * len(non_cancer_indices))))
    train_indices.update(random.sample(cancer_indices, int(0.6 * len(cancer_indices))))

    all_indices = non_cancer_indices
    all_indices.update(cancer_indices)

    test_indices = all_indices.difference(train_indices)

    training_data = torch.utils.data.Subset(data, list(train_indices))
    testing_data = torch.utils.data.Subset(data, list(test_indices))
    '''

    # train_loader = DataLoader(training_data, batch_size=10, shuffle=True)
    # test_loader = DataLoader(testing_data, batch_size=10)
    num_train = int(0.5 * len(data))
    num_val = len(data) - num_train
    training_data, testing_data = torch.utils.data.random_split(data, (num_train, num_val))
    train_loader = DataLoader(training_data, batch_size=5, shuffle=True)
    test_loader = DataLoader(testing_data, batch_size=5)

    # Hyper-parameters
    num_layers = 9
    lr = 0.000001
    final_lr = lr * 100
    weight_decay = 0.05
    num_epochs = 75

    softmax = False

    if softmax:
        cnn_type = CNN2
        best_model = 46
        loss_function = nn.CrossEntropyLoss().cuda(cuda_destination)
    else:
        cnn_type = CNN
        best_model = 1
        loss_function = nn.BCELoss().cuda(cuda_destination)

    for num_layers_to_freeze in range(8, num_layers):
        print("{} layers frozen".format(num_layers_to_freeze))
        model = cnn_type(cuda_destination)
        model.cuda(cuda_destination)
        model.load_state_dict(torch.load("/home/andrewg/PycharmProjects/assignments/predictions/models/{}.pt".format(
                                         best_model),
                                         map_location=device))
        '''
        for child_num in range(-1, -1 * (num_layers - num_layers_to_freeze + 1), -1):
            print("Reinitialized layer {}".format(num_layers + child_num + 1))
            nn.init.kaiming_normal_(list(model.children())[child_num].weight)

        if softmax:
            change_requires_grad(model, 6, False)
            nn.init.kaiming_normal_(list(model.children())[6].weight)

        else:
            change_requires_grad(model, num_layers_to_freeze, False)
        '''
        optimizer = AdaBound(model.parameters(), lr=lr, final_lr=final_lr, weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        
        _, conf_matrix, _, _, _, _ = train_model(train_loader, test_loader, model, num_epochs, optimizer,
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
            CNN2.visualize(images[i*20], model.class_activation_mapping(images[i*20].unsqueeze(0)))


