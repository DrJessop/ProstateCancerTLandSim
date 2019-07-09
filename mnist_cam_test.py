import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from adabound import AdaBound
from data_helpers import train_model
from models import MNIST_CNN


seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

cuda_destination = 0
ngpu = 1
device = torch.device("cuda:{}".format(cuda_destination) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])
                                            ]
                                            ))

mnist_loader = DataLoader(mnist_trainset, batch_size=200)
mnist_cnn = MNIST_CNN(0)
mnist_cnn.cuda(0)

training_data, testing_data = torch.utils.data.random_split(mnist_trainset, (int(0.8 * len(mnist_trainset)),
                                                                             int(0.2 * len(mnist_trainset))))
train_loader = DataLoader(training_data, batch_size=200)
test_loader = DataLoader(testing_data, batch_size=200)
optimizer = AdaBound(mnist_cnn.parameters(), lr=0.0001, final_lr=0.001, weight_decay=0.05)
loss_function = nn.CrossEntropyLoss().cuda(cuda_destination)
train_model(mnist_loader, test_loader, mnist_cnn, 40, optimizer,
            loss_function, softmax=True, show=True)
image = next(iter(train_loader))[0].cuda(0)
print(image.shape)
for i in range(10):
    print(MNIST_CNN.visualize(image[i], mnist_cnn.class_activation_mapping(image[i])))