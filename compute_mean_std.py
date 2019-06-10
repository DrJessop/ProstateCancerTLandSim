import SimpleITK as sitk
import os
import numpy as np

for modality in ["t2", "adc", "bval"]:

    train_dir = r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/{}".format(modality)
    test_dir = r"/home/andrewg/PycharmProjects/assignments/resampled_cropped/test/{}".format(modality)

    mean_tensor = np.zeros((3, 32, 32))

    train_directory_contents = os.listdir(train_dir)

    for image_file in train_directory_contents:
        image = sitk.GetArrayFromImage(sitk.ReadImage("{}/{}".format(train_dir, image_file)))
        mean_tensor = mean_tensor + image

    mean_tensor = mean_tensor / len(train_directory_contents)

    standard_deviation_tensor = np.zeros((3, 32, 32))

    for image_file in train_directory_contents:
        image = sitk.GetArrayFromImage(sitk.ReadImage("{}/{}".format(train_dir, image_file)))
        standard_deviation_tensor = standard_deviation_tensor + np.square(image - mean_tensor)

    standard_deviation_tensor = np.sqrt(standard_deviation_tensor / len(train_directory_contents))

    np.save("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/{}_mean_tensor".format(modality),
            mean_tensor)
    np.save("/home/andrewg/PycharmProjects/assignments/resampled_cropped/train/{}_std_tensor".format(modality),
            standard_deviation_tensor)


