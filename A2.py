import SimpleITK as sitk
import numpy as np
from A1 import create_patients
import os
import pandas as pd
from image_augmentation import rotation3d
import shutil
import matplotlib.pyplot as plt
import random
import pickle as pk


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
        resample.SetInterpolator(sitk.sitkCosineWindowedSinc)

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


def crop_from_center(images, ijk_coordinates, width, height, depth, i_offset=0, j_offset=0):
    """
    Helper function for image cropper and rotated crop that produces a crop of dimension width x height x depth,
    where the lesion is offset by i_offset (x dimension) and j_offset (y dimension)
    :param images: A list of size 3 tuples, where the elements in the tuple are t2, adc, and bval SITK images
                   respectively
    :param ijk_coordinates: The coordinates of the lesion
    :param width: Desired width of the crop
    :param height: Desired height of the crop
    :param depth: Desired depth of the crop
    :param i_offset: Desired offset in pixels away from the lesion in the x direction
    :param j_offset: Desired offset in pixels away from the lesion in the y direction
    :return: The newly created crop
    """
    crop = [image[(ijk_coordinates[idx][0] - i_offset) - width // 2: (ijk_coordinates[idx][0] - i_offset)
                  + int(np.ceil(width / 2)),
                  (ijk_coordinates[idx][1] - j_offset) - height // 2: (ijk_coordinates[idx][1] - j_offset)
                  + int(np.ceil(height / 2)),
                  ijk_coordinates[idx][2] - depth // 2: ijk_coordinates[idx][2]
                  + int(np.ceil(depth / 2))]
            for idx, image in enumerate(images)]
    return crop


def rotated_crop(patient_images, crop_width, crop_height, crop_depth, degrees, lps, ijk_values, show_result=False):
    """
    This is a helper function for image_cropper, and it returns a crop around a rotated image
    :param patient_image: The sitk image that is to be cropped
    :param crop_width: The desired width of the crop
    :param crop_height: The desired height of the crop
    :param crop_depth: The desired depth of the crop
    :param degrees: A list of all allowable degrees of rotation (gets converted to radians in the rotation3d function
                    which is called below)
    :param lps: The region of interest which will be the center of rotation
    :param ijk_values: A list of lists, where each list is the ijk values for each image's biopsy position
    :param show_result: Whether or not the user wants to see the first slice of the new results
    :return: The crop of the rotated image
    """

    degree = np.random.choice(degrees)
    rotated_patient_images = list(map(lambda patient: rotation3d(patient, degree, lps), patient_images))

    i_offset = np.random.randint(-7, 7)
    j_offset = np.random.randint(-7, 7)

    crop = crop_from_center(rotated_patient_images, ijk_values, crop_width, crop_height, crop_depth, i_offset=i_offset,
                            j_offset=j_offset)

    if show_result:
        for i in range(3):
            plt.imshow(sitk.GetArrayFromImage(rotated_patient_images[0])[0], cmap="gray")
            plt.imshow(sitk.GetArrayFromImage(crop[i])[0], cmap="gray")
            plt.show()
        input()
    return crop


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

    t2_resampled, adc_resampled, bval_resampled = resampled_images

    if num_crops_per_image < 1:
        print("Cannot have less than 1 crop for an image")
        exit()
    degrees = [5, 10, 15, 20, 25, 180]  # One of these is randomly chosen for every rotated crop
    crops = {}
    invalid_keys = set()
    for _, patient in findings_dataframe.iterrows():
        patient_id = patient["patient_id"]
        patient_images = [t2_resampled[int(patient_id[-4:])], adc_resampled[int(patient_id[-4:])],
                          bval_resampled[int(patient_id[-4:])]]
        if train:
            cancer_marker = int(patient["ClinSig"])  # 1 if cancer, else 0
        if '' in patient_images:  # One of the images is blank
            continue
        else:
            # Adds padding to each of the images
            patient_images = [padding.Execute(p_image) for p_image in patient_images]
            lps = [float(loc) for loc in patient["pos"].split(' ') if loc != '']

            # Convert lps to ijk for each of the images
            ijk_vals = [patient_images[idx].TransformPhysicalPointToIndex(lps) for idx in range(3)]

            # Below code makes a crop of dimensions crop_width x crop_height x crop_depth
            for crop_num in range(num_crops_per_image):
                if crop_num == 0:  # The first crop we want to guarantee has the biopsy position exactly in the center
                    crop = crop_from_center(patient_images, ijk_vals, crop_width, crop_height, crop_depth)
                else:
                    # Rotate the image, and then translate and crop
                    crop = rotated_crop(patient_images, crop_width, crop_height, crop_depth, degrees, lps, ijk_vals)
                sizes = [im.GetSize() for im in crop if im.GetSize() == (crop_width, crop_height, crop_depth)]
                if train:
                    if len(sizes) != 3:  # If not all of the image sizes are correct
                        print("Invalid image for patient {}".format(patient_id))
                        invalid_keys.add("{}_{}".format(patient_id, cancer_marker))
                        continue
                if train:
                    key = "{}_{}".format(patient_id, cancer_marker)
                else:
                    key = patient_id
                if key in crops.keys():
                    if train:
                        crops[key].append((crop, cancer_marker))
                    else:
                        crops[key].append(crop)
                else:
                    if train:
                        crops[key] = [(crop, cancer_marker)]
                    else:
                        crops[key] = [crop]

    for key in invalid_keys:
        crops.pop(key)
    return crops


def write_cropped_images(cropped_images, train=True):
    """
    This function writes the cropped images of modality 'modality' (ex. t2-weighted, bval, etc.)
    to the directory resampled_cropped
    :param cropped_images: A dictionary where the key is the patient number and the value is
                           a list of the crops around all the relevant fiducials
    :param train: Whether or not we are writing to the training or test set
    :return: None
    """

    if train:
        destination = r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/"
    else:
        destination = r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/test/"

    directory_contents = os.listdir(destination)
    for sub_directory in directory_contents:
        sub_directory_path = destination + sub_directory
        shutil.rmtree(sub_directory_path)
        os.mkdir(sub_directory_path)

    destination = destination + r"{}/{}_{}.nrrd" if train else destination + r"{}/{}.nrrd"

    patient_images = [patient_image for key in cropped_images.keys()
                      for patient_image in cropped_images[key]]
    for p_id in range(len(patient_images)):
        if train:
            patient_image, cancer = patient_images[p_id]
            sitk.WriteImage(patient_image[0], destination.format("t2", p_id, cancer))
            sitk.WriteImage(patient_image[1], destination.format("adc", p_id, cancer))
            sitk.WriteImage(patient_image[2], destination.format("bval", p_id, cancer))
        else:
            patient_image = patient_images[p_id]
            sitk.WriteImage(patient_image[0], destination.format("t2", p_id))
            sitk.WriteImage(patient_image[1], destination.format("adc", p_id))
            sitk.WriteImage(patient_image[2], destination.format("bval", p_id))
    return


def write_cropped_images_train_and_folds(cropped_images, num_crops, num_folds=5, fold_fraction=0.2):
    """
    This function writes all cropped images to a training directory (for each modality) and creates a list of hashmaps
    for folds. These maps ensure that there is a balanced distribution of cancer and non-cancer in each validation set
    as well as the training set used for prediction.
    :param cropped_images: A dictionary where the keys are the patient IDs, and the values are lists where each element
    is a list of length three (first element in that list is t2 image, and then adc and bval).
    :param num_crops: The number of crops for a given patient's image
    :param num_folds: The number of sets to be created
    :param fold_fraction: The amount of cancer patients to be within a fold's validation set
    :return: fold key and train key mappings (lists of hash functions which map to the correct patient data)
    """

    destination = r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/"

    directory_contents = os.listdir(destination)
    for sub_directory in directory_contents:
        sub_directory_path = destination + sub_directory
        shutil.rmtree(sub_directory_path)
        os.mkdir(sub_directory_path)

    destination = destination + r"{}/{}_{}.nrrd"

    patient_images = [(key, patient_image) for key in cropped_images.keys()
                      for patient_image in cropped_images[key]]

    for p_id in range(len(patient_images)):
        _, (patient_image, cancer_marker) = patient_images[p_id]
        sitk.WriteImage(patient_image[0], destination.format("t2", p_id, cancer_marker))
        sitk.WriteImage(patient_image[1], destination.format("adc", p_id, cancer_marker))
        sitk.WriteImage(patient_image[2], destination.format("bval", p_id, cancer_marker))

    patient_indices = set(range(len(patient_images) // num_crops))
    non_cancer_patients = {idx for idx in patient_indices if patient_images[idx * num_crops][0][-1] == '0'}
    cancer_patients = {idx for idx in patient_indices if patient_images[idx * num_crops][0][-1] == '1'}

    num_each_class_fold = int(fold_fraction * len(cancer_patients))

    fold_key_mappings = []
    train_key_mappings = []
    for k in range(num_folds):

        non_cancer_in_fold = random.sample(non_cancer_patients, num_each_class_fold)
        cancer_in_fold = random.sample(cancer_patients, num_each_class_fold)

        fold_set = set()
        fold_set.update(non_cancer_in_fold)
        fold_set.update(cancer_in_fold)

        out_of_fold = patient_indices.difference(fold_set)

        # Uses up all the cancer patients
        cancer_out_of_fold = {idx for idx in out_of_fold if patient_images[idx * num_crops][0][-1] == '1'}
        non_cancer_out_of_fold = random.sample(out_of_fold.difference(cancer_out_of_fold), len(cancer_out_of_fold))

        out_of_fold_set = set()
        out_of_fold_set.update(cancer_out_of_fold)
        out_of_fold_set.update(non_cancer_out_of_fold)

        # Prepare fold indices
        fold_image_indices = set()
        for key in fold_set:
            image_index = key * num_crops
            for pos in range(num_crops):
                fold_image_indices.add(image_index + pos)

        # Prepare train key indices
        out_of_fold_image_indices = set()
        for key in out_of_fold_set:
            image_index = key * num_crops
            for pos in range(num_crops):
                out_of_fold_image_indices.add(image_index + pos)

        fold_key_mapping = {}
        key = 0
        for fold_image_index in fold_image_indices:
            fold_key_mapping[key] = fold_image_index
            key += 1

        train_key_mapping = {}
        key = 0
        for train_image_index in out_of_fold_image_indices:
            train_key_mapping[key] = train_image_index
            key += 1

        fold_key_mappings.append(fold_key_mapping)
        train_key_mappings.append(train_key_mapping)

    return fold_key_mappings, train_key_mappings


def write_cropped_images_test(cropped_images):
    """
    This function writes the cropped testing images to the test folder (for each modality)
    :param cropped_images: A dictionary, where the key is the patient number, and the value is a list of length three,
    where the images in the list are t2, adc, and bval respectively
    :return: None
    """

    destination = r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/test/"

    directory_contents = os.listdir(destination)
    for sub_directory in directory_contents:
        sub_directory_path = destination + sub_directory
        shutil.rmtree(sub_directory_path)
        os.mkdir(sub_directory_path)

    destination = destination + r"{}/{}.nrrd"

    # Replace missing images with fake images
    df = pd.read_csv("/home/andrewg/PycharmProjects/assignments/ProstateX-TestLesionInformation/ProstateX-Findings-Test.csv")
    missing_keys = set(df["ProxID"]).difference(cropped_images.keys())
    for missing_key in missing_keys:
        num_random_images_to_add = len(df.ProxID[df.ProxID == missing_key])
        for rand_img in range(num_random_images_to_add):
            if missing_key in cropped_images.keys():
                cropped_images[missing_key].append([sitk.Image(32, 32, 3, sitk.sitkInt16)]*3)
            else:
                cropped_images[missing_key] = [[sitk.Image(32, 32, 3, sitk.sitkInt16)]*3]

    patient_images = [patient_image for key in sorted(cropped_images.keys())
                      for patient_image in cropped_images[key]]
    for p_id in range(len(patient_images)):
        patient_image = patient_images[p_id]
        sitk.WriteImage(patient_image[0], destination.format("t2", p_id))
        sitk.WriteImage(patient_image[1], destination.format("adc", p_id))
        sitk.WriteImage(patient_image[2], destination.format("bval", p_id))


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

    resampled_images = [t2, adc, bval]

    num_crops = 20

    cropped_images_train = image_cropper(findings_train, resampled_images, padding_filter, *desired_patch_dimensions,
                                         num_crops_per_image=num_crops, train=True)

    fold_key_mappings, train_key_mappings = write_cropped_images_train_and_folds(cropped_images_train,
                                                                                 num_crops=num_crops)
    with open("/home/andrewg/PycharmProjects/assignments/fold_key_mappings.pkl", 'wb') as output:
        pk.dump(fold_key_mappings, output, pk.HIGHEST_PROTOCOL)

    with open("/home/andrewg/PycharmProjects/assignments/train_key_mappings.pkl", 'wb') as output:
        pk.dump(train_key_mappings, output, pk.HIGHEST_PROTOCOL)

    cropped_images_test = image_cropper(findings_test, resampled_images, padding_filter, *desired_patch_dimensions,
                                        num_crops_per_image=1, train=False)

    write_cropped_images_test(cropped_images_test)

    print("Done")
