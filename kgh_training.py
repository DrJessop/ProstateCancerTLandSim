import torch
from models import CNN
import adabound

if __name__ == "__main__":
    cuda_destination = 1
    model = CNN(cuda_destination)
    ngpu = 1
    device = torch.device("cuda:{}".format(cuda_destination) if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    model.load_state_dict(torch.load("/home/andrewg/PycharmProjects/assignments/predictions/models/1.pt",
                                     map_location=device))
