import torch
from A3 import CNN, ProstateImages
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

model = CNN()
model.cuda()
model.load_state_dict(torch.load("/home/andrewg/PycharmProjects/assignments (copy)/predictions/models/1.pt"))
model.eval()

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
p_images_test = ProstateImages(modality="bval", train=False, device=device)
dataloader_test = DataLoader(p_images_test, batch_size=50, shuffle=False)

dummy_input = next(iter(dataloader_test))["image"][0].unsqueeze(0)
dummy_input.cuda()

writer = SummaryWriter("/home/andrewg/PycharmProjects/assignments/tensorboard_examples/graphs")

writer.add_graph(model, dummy_input)
writer.close()
