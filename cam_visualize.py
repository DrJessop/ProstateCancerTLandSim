import torch
from data_helpers import nrrd_to_tensor
from models import CNN2


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