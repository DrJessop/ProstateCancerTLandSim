import torch.utils.data
from models import CNN
from sklearn.metrics import auc, roc_curve
from data_helpers import KGHProstateImages, bootstrap_auc
import matplotlib.pyplot as plt

if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cuda_destination = 0
    ngpu = 1
    device = torch.device("cuda:{}".format(cuda_destination) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    model = CNN(cuda_destination=cuda_destination)
    model.load_state_dict(torch.load(
        "/home/andrewg/PycharmProjects/assignments/predictions/models/1.pt",
        map_location=device))
    model.cuda(cuda_destination)
    model.eval()
    data = KGHProstateImages(device)

    original_images = data[0]["image"][0].unsqueeze(0)
    target = torch.tensor(data[0]["cancer"]).unsqueeze(0)

    for idx in range(1, len(data)):
        next_image = data[idx]["image"][0].unsqueeze(0)
        original_images = torch.cat((original_images, next_image), 0)
        target = torch.cat((target, torch.tensor(data[idx]["cancer"]).unsqueeze(0)))

    original_images = original_images.to(device).float()
    class_vector = target.to(device).float()

    predictions = model(original_images).cpu().detach().numpy()
    targets = class_vector.cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot()
    bootstrap_auc(targets, predictions, ax, nsamples=500)
    plt.show()
    fpr, tpr, _ = roc_curve(targets, predictions, pos_label=1)
    print("The AUC is {}".format(auc(fpr, tpr)))



