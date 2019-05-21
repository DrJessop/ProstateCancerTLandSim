import SimpleITK as sitk
import numpy as np
from A1 import create_patients
import os
import pandas as pd
from image_augmentation import rotation3d
import matplotlib.pyplot as plt


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
        return [resample_image(mod_image, out_spacing) if mod_image != "" else ""
                for mod_image in modality]
    return [resample_image(mod_image, out_spacing)
            if mod_image != "" else "" for mod_image in modality]


def image_cropper(findings_dataframe, resampled_images, padding,
                  crop_width, crop_height, crop_depth, num_crops_per_image=1, train=True):
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
    :param num_crops_per_image: The number of crops desired for a given image
    :param train: Boolean, represents whether these are crops of the training or the test set
    :return: A list of cropped versions of the original re-sampled images
    """

    '''
    TODO: Instead of ofc, have a numcrops function, and then take a bunch of crops around the image and 
    append them to crops[patient_id]
    '''
    if num_crops_per_image < 1:
        print("Cannot have less than 1 crop for an image")
        exit()
    degrees = [5, 10, 15, 20, 25]
    crops = {}
    for _, patient in findings_dataframe.iterrows():
        patient_id = patient["patient_id"]
        patient_image = resampled_images[int(patient_id[-4:])]
        if train:
            cancer_marker = int(patient["ClinSig"])  # 1 if cancer, else 0
        if patient_image == '':
            continue
        else:
            patient_image = padding.Execute(patient_image)

            lps = [float(loc) for loc in patient["pos"].split(' ') if loc != '']
            i_coord, j_coord, k_coord = patient_image.TransformPhysicalPointToIndex(lps)  # Convert LPS to IJK
            # Below code makes a crop of dimensions crop_width x crop_height x crop_depth
            for crop_num in range(num_crops_per_image):
                if crop_num == 0:  # The first crop we want to guarantee has the ROI exactly in the center
                    crop = patient_image[i_coord - crop_width // 2: i_coord + int(np.ceil(crop_width / 2)),
                                         j_coord - crop_height // 2: j_coord + int(np.ceil(crop_height / 2)),
                                         k_coord - crop_depth // 2: k_coord + int(np.ceil(crop_depth / 2))]
                else:
                    dist_from_i_coord = np.random.choice((np.random.randint(-4, -1), np.random.randint(2, 5)))
                    dist_from_j_coord = np.random.choice((np.random.randint(-4, -1), np.random.randint(2, 5)))
                    dist_from_k_coord = np.random.choice((np.random.randint(-4, -1), np.random.randint(2, 5)))
                    i_sign = np.sign(dist_from_i_coord)
                    j_sign = np.sign(dist_from_j_coord)
                    k_sign = np.sign(dist_from_k_coord)

                    i_offset1 = crop_width // dist_from_i_coord
                    i_offset2 = i_sign * (crop_width - abs(i_offset1))

                    j_offset1 = crop_height // dist_from_j_coord
                    j_offset2 = j_sign * (crop_width - abs(j_offset1))

                    k_offset1 = crop_depth // dist_from_k_coord
                    k_offset2 = k_sign * (crop_depth - abs(k_offset1))

                    degree = np.random.choice(degrees)
                    rotated_patient_image = rotation3d(patient_image, degree, lps)
                    i_coord, j_coord, k_coord = rotated_patient_image.TransformPhysicalPointToIndex(lps)

                    crop = rotated_patient_image[i_coord - i_offset1: i_coord + i_offset2: i_sign,
                                                 j_coord - j_offset1: j_coord + j_offset2: j_sign,
                                                 k_coord - k_offset1: k_coord + k_offset2: k_sign]

                if crop.GetSize() != (crop_width, crop_height, crop_depth):
                    print("There seems to be a problem with patient id={}".format(patient_id))
                    continue
                if patient_id in crops.keys():
                    if train:
                        crops[patient_id].append((crop, cancer_marker))
                    else:
                        crops[patient_id].append(crop)
                else:
                    if train:
                        crops[patient_id] = [(crop, cancer_marker)]
                    else:
                        crops[patient_id] = [crop]
                if train and cancer_marker == 0:
                    break  # We do not need to generate multiple instances of non-cancer data
    return crops


def write_cropped_images(cropped_images, modality, train=True):
    """
    This function writes the cropped images of modality 'modality' (ex. t2-weighted, bval, etc.)
    to the directory resampled_cropped
    :param cropped_images: A dictionary where the key is the patient number and the value is
    a list of the crops around all the relevant fiducials
    :param modality: ex. t2, adc, bval
    :param train: Whether or not we are writing to the training or test set
    :return: None
    """

    if train:
        destination = \
            r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/{}/{}_{}_{}.nrrd".format(
                                                                            modality, "{}", "{}", "{}")
    else:
        destination = \
            r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/test/{}/{}_{}.nrrd".format(
                                                                            modality, "{}", "{}")
    for patient_number in cropped_images.keys():
        fid_number = 0
        for patient_image in cropped_images[patient_number]:
            if train:
                patient_image, cancer = patient_image
                sitk.WriteImage(patient_image, destination.format(patient_number, fid_number, cancer))
            else:
                sitk.WriteImage(patient_image, destination.format(patient_number, fid_number))
            fid_number += 1


def write_cropped_images_v2(cropped_images, modality, train=True):
    """
    This function writes the cropped images of modality 'modality' (ex. t2-weighted, bval, etc.)
    to the directory resampled_cropped
    :param cropped_images: A dictionary where the key is the patient number and the value is
    a list of the crops around all the relevant fiducials
    :param modality: ex. t2, adc, bval
    :param train: Whether or not we are writing to the training or test set
    :return: None
    """

    if train:
        destination = \
            r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/{}/{}_{}.nrrd".format(
                                                                            modality, "{}", "{}")
    else:
        destination = \
            r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/test/{}/{}.nrrd".format(
                                                                            modality, "{}")

    patient_images = [patient_image for key in cropped_images.keys() for patient_image in cropped_images[key]]
    for p_id in range(len(patient_images)):
        if train:
            patient_image, cancer = patient_images[p_id]
            sitk.WriteImage(patient_image, destination.format(p_id, cancer))
        else:
            sitk.WriteImage(patient_images[p_id], destination.format(p_id))


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
    findings_train = pd.read_csv(r"{}{}".format("/home/andrewg/PycharmProjects/assignments/",
                                 "ProstateX-TrainingLesionInformationv2/ProstateX-Findings-Train.csv"))

    findings_test = pd.read_csv(r"{}{}".format("/home/andrewg/PycharmProjects/assignments/",
                                                "ProstateX-TestLesionInformation/ProstateX-Findings-Test.csv"))

    new_columns = ["patient_id"]
    new_columns.extend(findings_train.columns[1:])  # The original first column was unnamed
    findings_train.columns = new_columns

    new_columns = ["patient_id"]
    new_columns.extend(findings_test.columns[1:])
    findings_test.columns = new_columns

    desired_patch_dimensions = (32, 32, 3)
    padding = (8, 8, 6)  # Necessary padding for fiducials that are on the border of an image
    padding_filter = sitk.ConstantPadImageFilter()
    padding_filter.SetPadLowerBound(padding)
    padding_filter.SetPadUpperBound(padding)
    padding_filter.SetConstant(0)
    cropped_images_t2_train = image_cropper(findings_train, t2, padding_filter, *desired_patch_dimensions,
                                            num_crops_per_image=3, train=True)
    cropped_images_adc_train = image_cropper(findings_train, adc, padding_filter, *desired_patch_dimensions,
                                             num_crops_per_image=3, train=True)
    cropped_images_bval_train = image_cropper(findings_train, bval, padding_filter, *desired_patch_dimensions,
                                              num_crops_per_image=3, train=True)

    cropped_images_t2_test = image_cropper(findings_test, t2, padding_filter, *desired_patch_dimensions,
                                           train=False)
    cropped_images_adc_test = image_cropper(findings_test, adc, padding_filter, *desired_patch_dimensions,
                                            train=False)
    cropped_images_bval_test = image_cropper(findings_test, bval, padding_filter, *desired_patch_dimensions,
                                             train=False)

    should_write_images = input("Would you like to write these cropped images to the " +
                                "re-sampled_cropped directory? y/n ")
    while should_write_images not in ['y', 'n']:
        print("Sorry, invalid response.")
        should_write_images = input("Would you like to write these cropped images to the " +
                                    "re-sampled_cropped directory? y/n ")
    if should_write_images == 'y':
        write_cropped_images_v2(cropped_images_t2_train, "t2", train=True)
        write_cropped_images_v2(cropped_images_adc_train, "adc", train=True)
        write_cropped_images_v2(cropped_images_bval_train, "bval", train=True)

        write_cropped_images_v2(cropped_images_t2_test, "t2", train=False)
        write_cropped_images_v2(cropped_images_adc_test, "adc", train=False)
        write_cropped_images_v2(cropped_images_bval_test, "bval", train=False)

    print("Done")


