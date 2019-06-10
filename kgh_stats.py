import pandas as pd
import pydicom
import os

top_directory = "/home/andrewg/PycharmProjects/Segmentation Parvin/MRIs"
top_directory_contents = os.listdir(top_directory)
data = []

for folder in top_directory_contents:
    if "wrong MRI imported" in folder:
        continue
    sub_directory = "{}/{}".format(top_directory, folder)
    sub_directory_contents = [sub_folder for sub_folder in os.listdir(sub_directory)
                              if os.path.isdir("{}/{}".format(sub_directory, sub_folder))]
    if sub_directory_contents[0][0] == 'e':
        # We want to choose the directory with the lower number since this directory contains the DCM files
        sub_directory_contents.sort()
        sub_directory = "{}/{}".format(sub_directory, sub_directory_contents[0])
        sub_directory_contents = os.listdir(sub_directory)
    for dicom_folder in sub_directory_contents:
        dicom_directory = "{}/{}".format(sub_directory, dicom_folder)
        dicom_folder_contents = next((d_folder for d_folder in os.listdir(dicom_directory)
                                      if "MRDC" in d_folder), None)
        if dicom_folder_contents:
            sample_file = "{}/{}".format(dicom_directory, dicom_folder_contents)
            sample_image = pydicom.read_file(sample_file)
            data.append({"SeriesDate": sample_image.SeriesDate,
                         "SeriesDescription": sample_image.SeriesDescription,
                         "Manufacturer": sample_image.Manufacturer,
                         "ManufacturerModelName": sample_image.ManufacturerModelName,
                         "MagneticFieldStrength": sample_image.MagneticFieldStrength,
                         "Modality": sample_image.Modality,
                         "TransmitCoilName": sample_image.TransmitCoilName,
                         "SliceThickness": sample_image.SliceThickness,
                         "PixelSpacing": sample_image.PixelSpacing,
                         "SeriesNumber": sample_image.SeriesNumber,
                         "PatientName": sample_image.PatientName})

df = pd.DataFrame(data=data)
df.to_csv("/home/andrewg/PycharmProjects/assignments/data/stats.csv", index=False)
