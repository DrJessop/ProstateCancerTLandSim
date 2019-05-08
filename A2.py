import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from A1 import create_patients
import os


t2_patient_006 = sitk.ReadImage(r"/home/andrewg/PycharmProjects/assignments/data/PROSTATEx/" +
                                r"ProstateX-0006/10-21-2011-MR prostaat kanker detectie NDmc MCAPRODETN-79408/" +
                                r"4-t2tsetra-98209/4-t2tsetra-98209.nrrd")

"""
# See the 11th (index 10) slice of the t2-weighted image for patient 0006 before equalization
plt.imshow(sitk.GetArrayFromImage(t2_patient_006)[10])
plt.show()

input("Press any key to continue...")
'''
# Apply histogram equalization to t2 image of patient 0006
hist_equalize_filter = sitk.AdaptiveHistogramEqualizationImageFilter()
t2_patient_006 = hist_equalize_filter.Execute(t2_patient_006)
t2_patient_006_arr = sitk.GetArrayFromImage(t2_patient_006)

# Plot the 11th (index 10) slice of the equalized t2-weighted image for patient 0006
plt.imshow(t2_patient_006_arr[10])
plt.show()

input("Press any key to continue...")
'''
# Take the 2D Fourier Transform of the slice
t2_patient_006_ft_arr = sitk.GetArrayFromImage(t2_patient_006)
t2_patient_006_ft_arr = np.fft.fft2(t2_patient_006_ft_arr[10])
fshift = np.fft.fftshift(t2_patient_006_ft_arr)
magnitude_spectrum = np.abs(fshift)
log_magnitude_spectrum = np.log(np.abs(fshift))
# plt.imshow(log_magnitude_spectrum)
# plt.title("Log Magnitude Spectrum of Slice 11 of Patient 0006")
# plt.show()

# input("Press any key to continue...")

inverse = np.fft.ifft2(fshift)
plt.imshow(np.abs(inverse))
plt.title("Inverse Fourier Transform after Setting Phase to 0")
plt.show()

input("Press any key to continue...")

# Create a 3x3 Laplacian filter and apply it in the spatial domain
laplacian_filter = np.matrix([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
laplacian_filter_fft = np.fft.fft2(laplacian_filter)
convolved_image_fft = signal.convolve2d(laplacian_filter_fft, fshift)
inverse_convolved_image_fft = np.fft.ifft2(convolved_image_fft)

plt.imshow(np.abs(inverse_convolved_image_fft))
plt.title("Prostate for slice 11 of Patient 0006 after Applying a Laplacian Filter")
plt.show()
"""
# Image Resampling


def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    """
    Retrieved this function from:
    https://www.programcreek.com/python/example/96383/SimpleITK.sitkNearestNeighbor
    :param itk_image: The image that we would like to resample
    :param out_spacing: The new spacing of the voxels we would like
    :param is_label: If True, use kNearestNeighbour interpolation, else use BSpline
    :return: The resampled image
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
        resample.SetInterpolator(sitk.sitkBSpline)

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
        return [resample_image(image, out_spacing) if image != "" else ""
                for image in modality]
    return [resample_image(image, out_spacing) if image != "" else "" for image in modality]


if __name__ == "__main__":
    patients = create_patients()
    t2 = [sitk.ReadImage(patients[patient_number]["t2"]) for patient_number in range(len(patients))]
    adc = [sitk.ReadImage(patients[patient_number]["adc"]) for patient_number in range(len(patients))]
    bval = [sitk.ReadImage(patients[patient_number]["bval"]) if patients[patient_number]["bval"] != ""
            else "" for patient_number in range(len(patients))]

    # Re-sampling all the images
    location = r"/home/andrewg/PycharmProjects/assignments/resampled/t2/t2_{}.nrrd"
    if not(os.listdir(r"/home/andrewg/PycharmProjects/assignments/resampled/t2")):
        t2[:] = resample_all_images(t2, out_spacing=(0.5, 0.5, 3))
        for patient_number, image in enumerate(t2):
            sitk.WriteImage(image, location.format(patient_number))
    else:
        t2 = [sitk.ReadImage(location.format(patient_number))
              for patient_number in range(len(patients))]

    location = r"/home/andrewg/PycharmProjects/assignments/resampled/adc/adc_{}.nrrd"
    if not(os.listdir(r"/home/andrewg/PycharmProjects/assignments/resampled/adc")):
        adc[:] = resample_all_images(adc, out_spacing=(2, 2, 3))
        for patient_number, image in enumerate(adc):
            sitk.WriteImage(image, location.format(patient_number))
    else:
        adc = [sitk.ReadImage(location.format(patient_number))
               for patient_number in range(len(patients))]

    location = r"/home/andrewg/PycharmProjects/assignments/resampled/bval/bval_{}.nrrd"
    if not(os.listdir(r"/home/andrewg/PycharmProjects/assignments/resampled/bval")):
        bval[:] = resample_all_images(bval, out_spacing=(2, 2, 3))
        for patient_number, image in enumerate(bval):
            if image != "":
                sitk.WriteImage(image, location.format(patient_number))
    else:
        def read_special_case(patient_number):
            try:
                return sitk.ReadImage(location.format(patient_number))
            except:
                return ""

        bval = [read_special_case(patient_number)
                for patient_number in range(len(patients))]

    plt.imshow(sitk.GetArrayFromImage(t2[0])[10])
    plt.show()



