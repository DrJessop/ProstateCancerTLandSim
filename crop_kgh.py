import os
import SimpleITK as sitk
from data_helpers import create_bval_kgh_patients
import shutil

if __name__ == "__main__":
    num_crops = 20
    crop_dim = (32, 32, 3)
    bval = create_bval_kgh_patients(num_crops=num_crops, crop_dim=crop_dim)
    crops_directory = "/home/andrewg/PycharmProjects/assignments/resampled_cropped/kgh"
    for key in bval.keys():
        patient_folder = "{}/{}".format(crops_directory, key)
        if key in os.listdir(crops_directory):
            shutil.rmtree(patient_folder)
        os.mkdir(patient_folder)
        for img_id in range(1, num_crops + 1):  # The first image is just the resampled un-cropped image
            sitk.WriteImage(bval[key][img_id], "{}/{}.nrrd".format(patient_folder, img_id - 1))
