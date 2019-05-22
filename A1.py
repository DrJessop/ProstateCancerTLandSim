import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt


def create_nrrd_files():
    """
    This function loops through the XChallenge directory structure and
    creates an nrrd file for each dicom file subdirectory
    :return: None
    """
    xchallenge_directory = r"/home/andrewg/PycharmProjects/assignments/data/PROSTATEx"

    reader = sitk.ImageSeriesReader()

    xchallenge_directory_contents = os.listdir(xchallenge_directory)  # All of the subdirectories in this directory

    num_patients = len(xchallenge_directory_contents)
    patient_counter = 1
    for patient_directory in xchallenge_directory_contents:
        print("On patient {} out of {}".format(patient_counter, num_patients))
        patient = "{}/{}".format(xchallenge_directory, patient_directory)
        patient = "{}/{}".format(patient, os.listdir(patient)[0])  # There is always one directory in the patient folder
        patient_contents = os.listdir(patient)
        for dicom_folder in patient_contents:
            directory_to_add_nrrd_file = "{}/{}".format(patient, dicom_folder)
            dicom_reader = reader.GetGDCMSeriesFileNames(directory_to_add_nrrd_file)
            reader.SetFileNames(dicom_reader)
            dicoms = reader.Execute()
            sitk.WriteImage(dicoms, "{}/{}".format(directory_to_add_nrrd_file, "{}.nrrd".format(dicom_folder)))
        patient_counter = patient_counter + 1


def find_nrrd(directory):
    """
    Given a directory, this function returns the nrrd file in the directory
    :param directory: string representing the location of the directory
    :return: string with location of the nrrd file
    """
    directory_contents = os.listdir(directory)
    file_with_extension = [file for file in directory_contents if ".nrrd" in file]
    return "{}/{}".format(directory, file_with_extension[0])


def get_nrrd_files(directory):
    """
    Given a patient directory, retrieves the t2, ADC, and BVAL nrrd files
    :param directory: The directory for a specific patient
    :return: Three lists, the first being the locations of the nrrd files for the t2 folders,
    the second for ADC, and the third for BVAL
    """
    t2, adc, bval = "", "", ""
    directory_contents = os.listdir(directory)
    for sub_directory in directory_contents:
        if "t2tsetra" in sub_directory:
            path = "{}/{}".format(directory, sub_directory)
            t2 = "{}".format(find_nrrd(path))
        elif "ADC" in sub_directory:
            path = "{}/{}".format(directory, sub_directory)
            adc = "{}".format(find_nrrd(path))
        elif "BVAL" in sub_directory:
            path = "{}/{}".format(directory, sub_directory)
            bval = "{}".format(find_nrrd(path))
    return t2, adc, bval


def create_patients():
    """
    Retrieves the t2, adc, and bval nrrd files for each patient and stores them in a dictionary
    :return: A dictionary of three different modalities for each patient
    """
    xchallenge_directory = r"/home/andrewg/PycharmProjects/assignments/data/PROSTATEx"
    xchallenge_directory_contents = os.listdir(xchallenge_directory)
    patient_dict = dict()
    for patient_directory in xchallenge_directory_contents:
        patient = "{}/{}".format(xchallenge_directory, patient_directory)
        patient_number = int(patient[-4:])
        patient = "{}/{}".format(patient, os.listdir(patient)[0])  # There is always one directory in the patient folder
        t2, adc, bval = get_nrrd_files(patient)  # Gets three different modalities for the patient
        patient_dict[patient_number] = {}
        current_patient = patient_dict[patient_number]
        current_patient["t2"] = t2
        current_patient["adc"] = adc
        current_patient["bval"] = bval
    return patient_dict


def create_spacing_histogram(spacial_info):
    """
    Creates a histogram for the spacial distribution across MRI images
    :param spacial_info: The spatial information
    :return: A dictionary with the spatial distribution of spacial_info
    """
    hist = {}
    for spacing in spacial_info:
        for dim in range(3):
            spacing[dim] = int(np.floor(spacing[dim] * 100)) / 100
        spacing_key = tuple(spacing)
        if spacing_key in hist:
            hist[spacing_key] += 1
        else:
            hist[spacing_key] = 1
    return hist


def create_size_histogram(size_info):
    """
    Creates a histogram for the size distribution across MRI images
    :param size_info: The size information
    :return: A dictionary with the size distribution of size_info
    """
    hist = {}
    for idx in range(len(size_info)):
        if size_info[idx] in hist.keys():
            hist[size_info[idx]] += 1
        else:
            hist[size_info[idx]] = 1
    return hist


def display_spatial_histogram(spatial_info, key):
    """
    Displays the histogram of spacing between voxels
    :param spatial_info: List of spacings for all images of type "key"
    :param key: The type of image
    :return: None
    """
    spatial_hist = create_spacing_histogram(spatial_info)
    spatial_hist = {key: val for (key, val) in spatial_hist.items() if val > 10}
    num_groups = len(spatial_hist)

    plt.bar(range(num_groups), list(spatial_hist.values()), align="center", width=0.2)
    plt.xticks(range(num_groups), list(spatial_hist.keys()), rotation="vertical")
    plt.title("Spatial distribution for {}".format(key))
    plt.show()
    input("Press any key to continue...")


def display_size_histogram(size_info, key):
    """
    Displays the histogram of image sizes
    :param size_info: List of sizes for all images of type "key"
    :param key: The type of image
    :return: None
    """
    size_hist = create_size_histogram(size_info)
    size_hist = {key: val for (key, val) in size_hist.items() if val > 10}
    num_groups = len(size_hist)

    plt.bar(range(num_groups), list(size_hist.values()), align="center", width=0.2)
    plt.xticks(range(num_groups), list(size_hist.keys()), rotation="vertical")
    plt.title("Size distribution for {}".format(key))
    plt.show()
    input("Press any key to continue...")


if __name__ == "__main__":
    patients = create_patients()

    t2_spacial_info = [list(sitk.ReadImage(patients[patient_number]["t2"]).GetSpacing())
                       for patient_number in range(len(patients))]

    bval_spacial_info = [list(sitk.ReadImage(patients[patient_number]["bval"]).GetSpacing())
                         for patient_number in range(len(patients))
                         if patients[patient_number]["bval"] != ""]

    adc_spacial_info = [list(sitk.ReadImage(patients[patient_number]["adc"]).GetSpacing())
                        for patient_number in range(len(patients))]

    t2_size_info = [sitk.ReadImage(patients[patient_number]["t2"]).GetSize()
                    for patient_number in range(len(patients))]

    bval_size_info = [sitk.ReadImage(patients[patient_number]["bval"]).GetSize()
                      for patient_number in range(len(patients))
                      if patients[patient_number]["bval"] != ""]

    adc_size_info = [sitk.ReadImage(patients[patient_number]["adc"]).GetSize()
                     for patient_number in range(len(patients))]

    # Plot the spatial distribution for the T2-weighted images
    display_spatial_histogram(t2_spacial_info, key="t2")

    # Plot the size distribution for the T2-weighted images
    display_size_histogram(t2_size_info, key="t2")

    # Plot the spatial distribution for the BVAL images
    display_spatial_histogram(bval_spacial_info, key="bval")

    # Plot the size distribution for the BVAL images
    display_size_histogram(bval_size_info, key="bval")

    # Plot the spatial distribution for the ADC images
    display_spatial_histogram(adc_spacial_info, key="adc")

    # Plot the size distribution for the ADC images
    display_size_histogram(adc_size_info, key="adc")

    # Plot the distribution in the intensity for the T2-weighted image for the first patient
    patient0_t2_file = patients[0]["t2"]
    t2_patient0 = sitk.GetArrayViewFromImage(sitk.ReadImage(patient0_t2_file))

    plt.hist(t2_patient0.flatten())
    plt.title("T2 Intensity Distribution For Patient 0000")
    plt.show()

    input("Press any key to continue...")

    # Plot the distribution in the intensity for the BVAL image for the first patient
    patient0_bval_file = patients[0]["bval"]
    bval_patient0 = sitk.GetArrayViewFromImage(sitk.ReadImage(patient0_bval_file))
    plt.hist(bval_patient0.flatten())
    plt.title("BVAL Intensity Distribution For Patient 0000")
    plt.show()

    input("Press any key to continue...")

    # Plot the distribution in the intensity for the ADC image for the first patient
    patient0_adc_file = patients[0]["adc"]
    adc_patient0 = sitk.GetArrayViewFromImage(sitk.ReadImage(patient0_adc_file))
    plt.hist(adc_patient0.flatten())
    plt.title("ADC Intensity Distribution For Patient 0000")
    plt.show()

    # Attempt to open mhd file
    ktrans_006 = r"/home/andrewg/PycharmProjects/assignments/KTrans/ProstateXKtrains-train-fixed/ProstateX-0006/" + \
                 r"ProstateX-0006-Ktrans.mhd"
    ktrans_image = sitk.ReadImage(ktrans_006)
    ktrans_image = sitk.GetArrayViewFromImage(ktrans_image)
    plt.imshow(ktrans_image[10])
    plt.show()
