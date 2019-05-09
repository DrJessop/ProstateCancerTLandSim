import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from A1 import create_patients
import os
import pandas as pd


t2_patient_006 = sitk.ReadImage(r"/home/andrewg/PycharmProjects/assignments/data/PROSTATEx/" +
                                r"ProstateX-0006/10-21-2011-MR prostaat kanker detectie NDmc MCAPRODETN-79408/" +
                                r"4-t2tsetra-98209/4-t2tsetra-98209.nrrd")

# Image Resampling
def resample_image(itk_image, out_spacing, is_label=False):
    """
    Retrieved this function from:
    https://www.programcreek.com/python/example/96383/SimpleITK.sitkNearestNeighbor
    :param itk_image: The image that we would like to resample
    :param out_spacing: The new spacing of the voxels we would like
    :param is_label: If True, use kNearestNeighbour interpolation, else use BSpline
    :return: The re-sampled image
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0]*(original_spacing[0]/out_spacing[0]))),
                int(np.round(original_size[1]*(original_spacing[1]/out_spacing[1]))),
                int(np.round(original_size[2]*(original_spacing[2]/out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def resample_all_images(modality, out_spacing, some_missing=False):
    """
    This function returns a list of re-sampled images for a given modality and desired spacing
    :param modality: ex. t2, adc, bval, etc.
    :param out_spacing: The desired spacing of the images
    :param some_missing: If an image may be missing, this may be set to True to handle the case
    of a missing image
    :return: Re-sampled images
    """
    if some_missing:
        return [resample_image(image, out_spacing) if image != "" else ""
                for image in modality]
    return [resample_image(image, out_spacing) if image != "" else "" for image in modality]


def image_cropper(findings_dataframe, resampled_images, padding,
                  crop_width, crop_height, crop_depth):
    """
    Given a dataframe with the findings of cancer, a list of images, and a desired width, height,
    and depth, this function returns a set of cropped versions of the original images of dimension
    crop_width x crop_height x crop_depth
    :param findings_dataframe: A pandas dataframe containing the LPS coordinates of the cancer
    :param resampled_images: A list of images that have been resampled to all have the same
    spacing
    :param padding: 0-Padding in the i,j,k directions
    :param crop_width: The desired width of a patch
    :param crop_height: The desired height of a patch
    :param crop_depth: The desired depth of a patch
    :return: A list of cropped versions of the original re-sampled images
    """

    crops = []
    for idx, patient in findings_dataframe.iterrows():
        patient_id = patient["patient_id"]
        patient_image = resampled_images[int(patient_id[-4:])]
        if patient_image == '':
            crops.append((patient_id, ''))
        else:
            patient_image = padding.Execute(patient_image)
            lps = [float(loc) for loc in patient["pos"].split(' ') if loc != '']
            i_coord, j_coord, k_coord = patient_image.TransformPhysicalPointToIndex(lps)

            # Below code makes a crop of dimensions crop_width x crop_height x crop_depth
            crop = patient_image[i_coord - crop_width//2: i_coord + int(np.ceil(crop_width/2)),
                                 j_coord - crop_height//2: j_coord + int(np.ceil(crop_height/2)),
                                 k_coord - crop_depth//2: k_coord + int(np.ceil(crop_depth/2))]

            crops.append((patient_id, crop))

    return crops


def write_cropped_images(cropped_images, modality, patient_list):
    """

    :param cropped_images:
    :param modality:
    :param patient_list:
    :return:
    """

    destination = \
        r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/{}/{}.nrrd".format(
                                                                                modality, "{}")
    for patient_number, image in enumerate(cropped_images):
        if image != '':
            sitk.WriteImage(image, destination.format(
                patient_list[patient_number][modality].split('/')[7]))



if __name__ == "__main__":
    patients = create_patients()
    t2 = [sitk.ReadImage(patients[patient_number]["t2"]) for patient_number in range(len(patients))]
    adc = [sitk.ReadImage(patients[patient_number]["adc"]) for patient_number in range(len(patients))]
    bval = [sitk.ReadImage(patients[patient_number]["bval"]) if patients[patient_number]["bval"] != ""
            else "" for patient_number in range(len(patients))]

    # Re-sampling all the images
    location = r"/home/andrewg/PycharmProjects/assignments/resampled/t2/{}.nrrd"
    if not(os.listdir(r"/home/andrewg/PycharmProjects/assignments/resampled/t2")):
        t2[:] = resample_all_images(t2, out_spacing=(0.5, 0.5, 3))
        for patient_number, image in enumerate(t2):
            # The 7th index is the position in the path which specifies which patient we are on
            sitk.WriteImage(image,
                            location.format(patients[patient_number]["t2"].split('/')[7]))
    else:
        t2 = [sitk.ReadImage(location.format(patients[patient_number]["t2"].split('/')[7]))
              for patient_number in range(len(patients))]

    location = r"/home/andrewg/PycharmProjects/assignments/resampled/adc/{}.nrrd"
    if not(os.listdir(r"/home/andrewg/PycharmProjects/assignments/resampled/adc")):
        adc[:] = resample_all_images(adc, out_spacing=(2, 2, 3))
        for patient_number, image in enumerate(adc):
            sitk.WriteImage(image,
                            location.format(patients[patient_number]["adc"].split('/')[7]))
    else:
        adc = [sitk.ReadImage(location.format(patients[patient_number]["adc"].split('/')[7]))
               for patient_number in range(len(patients))]

    location = r"/home/andrewg/PycharmProjects/assignments/resampled/bval/{}.nrrd"
    if not(os.listdir(r"/home/andrewg/PycharmProjects/assignments/resampled/bval")):
        bval[:] = resample_all_images(bval, out_spacing=(2, 2, 3))
        for patient_number, image in enumerate(bval):
            if image != "":
                sitk.WriteImage(image,
                                location.format(patients[patient_number]["bval"].split('/')[7]))
    else:
        def read_special_case(patient_number):
            try:
                return sitk.ReadImage(
                    location.format(patients[patient_number]["bval"].split('/')[7]))
            except:
                return ""

        bval = [read_special_case(patient_number)
                for patient_number in range(len(patients))]

    # Open up the findings csv
    findings = pd.read_csv(r"/home/andrewg/PycharmProjects/assignments/" +
                           "ProstateX-TrainingLesionInformationv2/ProstateX-Findings-Train.csv")

    new_columns = ["patient_id"]
    new_columns.extend(findings.columns[1:])  # The original first column was unnamed
    findings.columns = new_columns

    desired_patch_dimensions = (32, 32, 3)
    padding = (4, 4, 5)
    padding_filter = sitk.ConstantPadImageFilter()
    padding_filter.SetPadLowerBound(padding)
    padding_filter.SetPadUpperBound(padding)
    padding_filter.SetConstant(0)
    cropped_images_t2 = image_cropper(findings, t2, padding_filter, *desired_patch_dimensions)
    # cropped_images_adc = image_cropper(findings, adc, *desired_patch_dimensions)
    # cropped_images_bval = image_cropper(findings, bval, *desired_patch_dimensions)
    '''
    write_cropped_images(cropped_images_t2, "t2", patients)
    write_cropped_images(cropped_images_adc, "adc", patients)
    write_cropped_images(cropped_images_bval, "bval", patients)
    '''


