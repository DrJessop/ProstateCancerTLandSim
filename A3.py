import os
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
        r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/{}".format(modality)
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
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((width//2, height//2, depth//2))


def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkBSpline
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def rotation(image, degrees, axis1, axis2, dim=3):
    """

    :param image:
    :param degrees:
    :param axis1:
    :param axis2:
    :param dim:
    :return:
    """
    affine = sitk.AffineTransform(dim)
    affine.SetCenter(get_center(image))
    radians = np.pi * degrees / 180
    affine.Rotate(axis1, axis2, angle=radians)
    return resample(image, affine)


def data_augmentation(small_class, modality, amount_needed):

    destination = r"{}/{}".format("/home/andrewg/PycharmProjects/assignments",
                                  "resampled_cropped_normalized_augmented")


    return


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

    print(num_cancer, num_non_cancer)

    sizes = [num_cancer, num_non_cancer]
    colors = ["yellowgreen", "lightblue"]
    explode = (0, 0.1)  # explode 2nd slice

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct="%1.1f%%", shadow=True, startangle=140)

    plt.axis("equal")
    plt.title("Pie chart of cancer percentage vs non-cancer")
    plt.show()

    # transform_45_degrees =


