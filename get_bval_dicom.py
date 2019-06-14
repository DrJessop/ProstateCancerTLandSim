import os
import SimpleITK as sitk
import pydicom
import shutil
from zipfile import ZipFile

kgh_data_dir = "/home/andrewg/PycharmProjects/Segmentation Parvin/MRIs"
kgh_data_destination = "/home/andrewg/PycharmProjects/assignments/data/KGHData/{}"
kgh_data_contents = os.listdir(kgh_data_dir)

reader = sitk.ImageSeriesReader()

for mri_image in kgh_data_contents:

    # These folders are useless for analysis
    if "wrong MRI imported" in mri_image:
        continue

    mri_image_dir = "{}/{}".format(kgh_data_dir, mri_image)

    mri_image_contents = os.listdir(mri_image_dir)

    mrb_file = next((f for f in mri_image_contents if "mrb" in f), None)
    if mrb_file:
        with ZipFile("{}/{}".format(mri_image_dir, mrb_file), "r") as zf:
            fiducial_file = next(f for f in zf.namelist() if "F.fcsv" in f)
            source_fid = zf.open(fiducial_file)
            with open("/home/andrewg/PycharmProjects/assignments/data/KGHData/fiducials/{}".format(mri_image),
                              "wb") as target_fid:
                with source_fid, target_fid:
                    shutil.copyfileobj(source_fid, target_fid)
            source_fid.close()

    mri_image_contents = [sub_folder for sub_folder in mri_image_contents
                          if os.path.isdir("{}/{}".format(mri_image_dir, sub_folder))]
    if mri_image_contents[0][0] == 'e':
        # We want to choose the directory with the lower number since this directory contains the DCM files
        mri_image_contents.sort()
        mri_image_dir = "{}/{}".format(mri_image_dir, mri_image_contents[0])
        mri_image_contents = os.listdir(mri_image_dir)
    target_folders = [folder for folder in mri_image_contents if folder[0] == 's']
    if target_folders:
        destination = kgh_data_destination.format(mri_image)
        try:
            os.mkdir(destination)
        except:
            shutil.rmtree(destination)
            os.mkdir(destination)
        for target_folder in target_folders:
            source = "{}/{}".format(mri_image_dir, target_folder)
            target = "{}/{}".format(destination, target_folder)

            # Skips the header and non-dicom files, sentinel None used in case no MRDC file in directory
            dicom_files = [f for f in os.listdir(source) if "MRDC" in f]
            bval_dicom_dict = dict()
            max_bval = 0
            bval_dicom_dict[max_bval] = list()
            for dicom_file in dicom_files:
                file_name = "{}/{}".format(source, dicom_file)
                metadata = pydicom.read_file(file_name)
                series_description = metadata.SeriesDescription
                if "DIFF B" in series_description and "ADC" not in series_description:
                    file_bval = int(metadata.SequenceName.split('b')[1].split('t')[0])
                    if file_bval > max_bval:
                        max_bval = file_bval
                        bval_dicom_dict[max_bval] = list()
                    if file_bval == max_bval:
                        bval_dicom_dict[max_bval].append((file_name.rsplit('/', 1)[1], metadata))

            if bval_dicom_dict[max_bval]:
                os.system("mkdir {}".format(target))
                for dicom_file, metadata in bval_dicom_dict[max_bval]:
                    file_location = "{}/{}".format(target, dicom_file)
                    pydicom.write_file(file_location, metadata)
                dicom_reader = reader.GetGDCMSeriesFileNames(target)
                reader.SetFileNames(dicom_reader)
                nrrd = reader.Execute()
                sitk.WriteImage(nrrd, "{}/{}.nrrd".format(destination, mri_image))
