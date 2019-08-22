import torch.utils.data
from models import CNN, CNN2
from sklearn.metrics import auc, roc_curve
from data_helpers import KGHProstateImages, bootstrap_auc
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cuda_destination = 0
    ngpu = 1
    device = torch.device("cuda:{}".format(cuda_destination) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    model = CNN2(cuda_destination=cuda_destination)
    model.load_state_dict(torch.load(
        "/home/andrewg/PycharmProjects/assignments/predictions/models/bval/CNN2/46.pt",
        map_location=device))
    model.cuda(cuda_destination)
    model.eval()
    data = KGHProstateImages(device, modality="t2")

    original_images = data[0]["image"][0].unsqueeze(0)
    indices = [data[0]["index"]]
    target = torch.tensor(data[0]["cancer"]).unsqueeze(0)

    for idx in range(1, len(data)):
        next_image = data[idx]["image"][0].unsqueeze(0)
        indices.append(data[idx]["index"])
        original_images = torch.cat((original_images, next_image), 0)
        target = torch.cat((target, torch.tensor(data[idx]["cancer"]).unsqueeze(0)))

    original_images = original_images.to(device).float()
    class_vector = target.to(device).float()

    predictions = model(original_images).cpu().detach().numpy()
    predictions = np.array([prediction[1] for prediction in predictions])
    targets = class_vector.cpu().detach().numpy()

    '''
    if isinstance(model, CNN2):
        for idx in range(60, 70):
            print(indices[idx], class_vector[idx], predictions[idx])
            chosen_image = original_images[idx]
            mapping = model.class_activation_mapping(chosen_image)
            plt.imshow(chosen_image.cpu().numpy()[1], cmap="gray", interpolation="bilinear")
            plt.axis("off")
            plt.show()
            CNN2.visualize(chosen_image, mapping)
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    bootstrap_auc(targets, predictions, ax, nsamples=500)
    plt.show()
    fpr, tpr, _ = roc_curve(targets, predictions, pos_label=1)
    print("The AUC is {}".format(auc(fpr, tpr)))



