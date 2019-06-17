import os
import SimpleITK as sitk
from data_helpers import resample_image, crop_from_center, create_bval_kgh_patients


if __name__ == "__main__":
    num_crops = 20
    bval = create_bval_kgh_patients(num_crops=num_crops)
    crops_directory = "/home/andrewg/PycharmProjects/assignments/resampled_cropped/kgh/{}.nrrd"
    for key in bval.keys():
        sitk.WriteImage(bval[key], crops_directory.format(key))
