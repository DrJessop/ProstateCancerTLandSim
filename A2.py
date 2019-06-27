import SimpleITK as sitk
from A1 import create_patients
import os
import pandas as pd
import shutil
from data_helpers import image_cropper, resample_all_images, write_cropped_images_train_and_folds
import pickle as pk


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
    print("Starting...")
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

    with open("/home/andrewg/PycharmProjects/assignments/fold_key_mappings2.pkl", 'wb') as output:
        pk.dump(fold_key_mappings, output, pk.HIGHEST_PROTOCOL)

    with open("/home/andrewg/PycharmProjects/assignments/train_key_mappings2.pkl", 'wb') as output:
        pk.dump(train_key_mappings, output, pk.HIGHEST_PROTOCOL)

    cropped_images_test = image_cropper(findings_test, resampled_images, padding_filter, *desired_patch_dimensions,
                                        num_crops_per_image=1, train=False)

    write_cropped_images_test(cropped_images_test)

    print("Done")
