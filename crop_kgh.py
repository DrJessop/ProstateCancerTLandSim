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
        for crop_num in range(num_crops):
            sitk.WriteImage(bval[key][crop_num], "{}/{}.nrrd".format(patient_folder, crop_num))
