import os
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils


def normalize_data(data):
    """
    This function normalizes images such that the mean is zero. Does so in-place.
    :param data: A double-dictionary of patients where the first key is the patient number
    and the second key is the fiducial number
    :return: None
    """
    normalize_image_filter = sitk.NormalizeImageFilter()
    for patient_number in data.keys():
        for fiducial_number in data[patient_number]:
            data[patient_number][fiducial_number] = normalize_image_filter.Execute(
                                                        data[patient_number][fiducial_number]
            )


def read_cropped_images(modality):
    """
    This function reads in images of a certain modality and stores them in the dictionary
    cropped_images
    :param modality: ex. t2, adc, bval, etc.
    :return: A dictionary where the first key is a patient number and the second key is the
    fiducial number (with 0 indexing)
    """
    cropped_images = {}
    destination = \
        r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/{}".format(modality)
    start_position_patient_number = -11
    end_position_patient_number = -8
    fiducial_number_pos = -6
    image_dir = os.listdir(destination)
    for image_file_name in image_dir:
        image = sitk.ReadImage("{}/{}".format(destination, image_file_name))
        patient_number = int(image_file_name[start_position_patient_number:
                                             end_position_patient_number + 1])
        fiducial_number = int(image_file_name[fiducial_number_pos])

        if patient_number not in cropped_images.keys():
            cropped_images[patient_number] = {}

        cropped_images[patient_number][fiducial_number] = image

    return cropped_images


def get_center(img):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                              int(np.ceil(height/2)),
                                              int(np.ceil(depth/2))))


def resample(image, transform):
    """
    This function resamples (updates) an image using a specified transform
    :param image: The sitk image we are trying to transform
    :param transform: An sitk transform (ex. resizing, rotation, etc.
    :return: The transformed sitk image
    """
    reference_image = image
    interpolator = sitk.sitkBSpline
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def rotation(image, degrees, axis1, axis2):
    """
    This function rotates an image by "degrees" degrees
    :param image: An sitk image
    :param degrees: The number of degrees we are trying to rotate the image by
    :param axis1:
    :param axis2:
    :return:
    """
    affine = sitk.AffineTransform(3)
    affine.SetCenter(get_center(image))
    radians = np.pi * degrees / 180
    affine.Rotate(axis1, axis2, angle=radians)
    return resample(image, affine)


def deg_to_rad(degs):
    return [np.pi * deg / 180 for deg in degs]


def rotation3(image, theta_x, theta_y, theta_z):
    # theta_x, theta_y, theta_z = deg_to_rad()
    theta_x = np.pi * theta_x / 180
    theta_y = np.pi * theta_y / 180
    theta_z = np.pi * theta_z / 180
    euler_transform = sitk.Euler3DTransform()
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)
    # print(euler_transform.GetCenter())
    euler_transform.SetRotation(theta_x, theta_y, theta_z)
    # euler_transform.SetTranslation((0, 2.5, 0))
    return resample(image, euler_transform)


def data_augmentation(small_class, modality, amount_needed):

    destination = r"{}/{}".format("/home/andrewg/PycharmProjects/assignments",
                                  "resampled_cropped_normalized_augmented")


    return


def single_number_to_four_digit(num):
    return ''.join(['0' for _ in range(3 - int(math.log(num, 10)))]) + str(num)


class ProstateImages(Dataset):
    def __init__(self, modality, train=True):
        self.modality = modality
        self.train = train

    def __len__(self):
        pass

    def __getitem__(self, patient_id, fiducial_number):
        if self.train:
            path = "{}/{}".format(
                "/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/",
                "ProstateX-{}_{}_{}.nrrd".format(single_number_to_four_digit(patient_id),
                                                 fiducial_number)
            )
            try:
                final_path = path.format(0)
                image = sitk.ReadImage(path)
            except:
                final_path = path.format(1)
                image = sitk.ReadImage(path)
            cancer_label = int(final_path.split('.')[0][-1])  # The last digit in the file name specifies cancer/non-cancer
            output = {"image": image, "cancer": cancer_label}

        else:
            path = "{}/{}".format(
                "/home/andrewg/PycharmProjects/assignments/resampled_cropped/test",
                "ProstateX-{}_{}.nrrd".format(single_number_to_four_digit(patient_id),
                                              fiducial_number)
            )
            image = sitk.ReadImage(path)
            output = {"image": image, "cancer": None}

        return output


if __name__ == "__main__":

    t2 = read_cropped_images("t2")
    adc = read_cropped_images("adc")
    bval = read_cropped_images("bval")

    normalize_data(t2)
    normalize_data(adc)
    normalize_data(bval)

    findings_df = pd.read_csv(r"/home/andrewg/PycharmProjects/assignments/" +
                              "ProstateX-TrainingLesionInformationv2/ProstateX-Findings-Train.csv")
    findings_df.ClinSig.apply(lambda clin_sig: int(clin_sig)).hist()
    plt.title("Histogram of cancer (1) vs non-cancer (0)")
    plt.show()

    input("Press enter to continue...")

    labels = "Cancer", "Non-Cancer"
    num_cancer = sum(findings_df.ClinSig)
    num_non_cancer = len(findings_df) - num_cancer

    sizes = [num_cancer, num_non_cancer]
    colors = ["yellowgreen", "lightblue"]
    explode = (0, 0.1)  # explode 2nd slice

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct="%1.1f%%", shadow=True, startangle=140)

    plt.axis("equal")
    plt.title("Pie chart of cancer percentage vs non-cancer")
    plt.show()

    # Create a training and validation set

    # img = sitk.ReadImage(r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/t2/ProstateX-0000_0.nrrd")
    # img_rot = rotation3(img, 0, 0, 0)
    # img_arr = sitk.GetArrayFromImage(img_rot)
    # img_arr = np.swapaxes(img_arr,0,2)[:,:,1]
    # plt.imshow(img_arr, cmap="gray"); plt.show()



