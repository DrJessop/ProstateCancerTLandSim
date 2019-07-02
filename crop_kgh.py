import os
import SimpleITK as sitk
from data_helpers import create_kgh_patient_crops
import shutil

if __name__ == "__main__":
    num_crops = 20
    crop_dim = (32, 32, 3)
    crops = create_kgh_patient_crops(num_crops=num_crops, crop_dim=crop_dim)
    crops_directory = "/home/andrewg/PycharmProjects/assignments/resampled_cropped/kgh"
    modalities = ["bval", "adc", "t2"]
    matching_filter = sitk.HistogramMatchingImageFilter()
    for crop_dict, modality in zip(crops, modalities):
        modality_directory = "{}/{}".format(crops_directory, modality)
        for key in crop_dict.keys():
            patient_folder = "{}/{}".format(modality_directory, key)
            if key in os.listdir(modality_directory):
                shutil.rmtree(patient_folder)
            os.mkdir(patient_folder)
            reference_image = crop_dict[key][1]  # The first image is just the resampled un-cropped image
            sitk.WriteImage(reference_image, "{}/{}.nrrd".format(patient_folder, 0))
            for img_id in range(2, num_crops + 1):
                crop_dict[key][img_id] = matching_filter.Execute(crop_dict[key][img_id], reference_image)
                sitk.WriteImage(crop_dict[key][img_id], "{}/{}.nrrd".format(patient_folder, img_id - 1))
