import torch
from data_helpers import nrrd_to_tensor
from models import CNN2
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter

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
model.eval()

im_num = "400_1.nrrd"
im3t_bval = nrrd_to_tensor("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/bval/{}".format(im_num))
im3t_adc = nrrd_to_tensor("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/adc/{}".format(im_num))
im3t_t2 = sitk.ReadImage("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/t2/{}".format(im_num))
# im3t_t2 = resample_image(im3t_t2, (2, 2, 3))
im3t_t2 = torch.from_numpy(sitk.GetArrayFromImage(im3t_t2).astype(np.float64))

im_num = "PCAD_005_tz"
im1_5t_bval = nrrd_to_tensor("/home/andrewg/PycharmProjects/assignments/resampled_cropped/kgh/bval/{}/0.nrrd".format(
                                                                                                                im_num))
im1_5t_adc = nrrd_to_tensor("/home/andrewg/PycharmProjects/assignments/resampled_cropped/kgh/adc/{}/0.nrrd".format(
                                                                                                                im_num))
im1_5t_t2 = nrrd_to_tensor("/home/andrewg/PycharmProjects/assignments/resampled_cropped/kgh/t2/{}/0.nrrd".format(
                                                                                                                im_num))


images3t = [im3t_bval, im3t_adc, im3t_t2]
images1_5t = [im1_5t_bval, im1_5t_adc, im1_5t_t2]

f, axarr = plt.subplots(2, 4, figsize=(21, 8))

for row in range(2):
    for col in range(4):
        axarr[row, col].axis("off")

modalities = ["BVAL", "ADC", "T2"]
for idx in range(3):
    im1 = gaussian_filter(images3t[idx].numpy()[1], sigma=0.5)
    im2 = gaussian_filter(images1_5t[idx].numpy()[1], sigma=0.5)
    axarr[0, idx].imshow(im1, interpolation="bilinear", cmap="gray")
    axarr[1, idx].imshow(im2, interpolation="bilinear", cmap="gray")
    axarr[0, idx].title.set_text(modalities[idx])
    axarr[0, idx].title.set_fontsize(30)

axarr[0, 3].title.set_text("CAM")
axarr[0, 3].title.set_fontsize(30)

heatmap3t = model.class_activation_mapping(im3t_bval.to(device).float())
heatmap3t = (heatmap3t[0][0] - heatmap3t[0][0].min()) / (heatmap3t[0][0].max() - heatmap3t[0][0].min())
heatmap1_5t = model.class_activation_mapping(im1_5t_bval.to(device).float())
heatmap1_5t = (heatmap1_5t[0][0] - heatmap1_5t[0][0].min()) / (heatmap1_5t[0][0].max() - heatmap1_5t[0][0].min())
axarr[0, 3].imshow(images3t[2].numpy()[1], interpolation="bilinear", cmap="gray")
axarr[1, 3].imshow(images1_5t[2].numpy()[1], interpolation="bilinear", cmap="gray")

heat1 = axarr[0, 3].imshow(heatmap3t.detach().cpu().numpy(), interpolation="bilinear", cmap="jet", alpha=0.6)
heat2 = axarr[1, 3].imshow(heatmap1_5t.detach().cpu().numpy(), interpolation="bilinear", cmap="jet", alpha=0.6)

cbar1 = plt.colorbar(heat1, ax=axarr[0, 3], format="%.1f")
cbar1.solids.set_edgecolor("face")
cbar1.solids.set_alpha(1)

cbar2 = plt.colorbar(heat2, ax=axarr[1, 3], format="%.1f")
cbar2.solids.set_edgecolor("face")
cbar2.solids.set_alpha(1)

plt.text(-160, -22, "3.0 T", fontsize=30)
plt.text(-160, 18, "1.5 T", fontsize=30)

center = heatmap3t.shape[0] // 2

axarr[0, 2].annotate("", xy=(center, center), xytext=(1.5*center, 0.5*center),
                     arrowprops=dict(facecolor='red', shrink=0.025), color="red")
axarr[1, 2].annotate("", xy=(center, center), xytext=(1.5*center, 0.5*center),
                     arrowprops=dict(facecolor='red', shrink=0.025), color="red")

plt.savefig("contrast_and_compare.eps", bbox_inches="tight")
# plt.show()
