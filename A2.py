import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


t2_patient_006 = sitk.ReadImage(r"/home/andrewg/PycharmProjects/assignments/data/PROSTATEx/" +
                                r"ProstateX-0006/10-21-2011-MR prostaat kanker detectie NDmc MCAPRODETN-79408/" +
                                r"4-t2tsetra-98209/4-t2tsetra-98209.nrrd")

# See the 11th (index 10) slice of the t2-weighted image for patient 0006 before equalization
plt.imshow(sitk.GetArrayFromImage(t2_patient_006)[10])
plt.show()

input("Press any key to continue...")

# Apply histogram equalization to t2 image of patient 0006
hist_equalize_filter = sitk.AdaptiveHistogramEqualizationImageFilter()
t2_patient_006 = hist_equalize_filter.Execute(t2_patient_006)
t2_patient_006_arr = sitk.GetArrayFromImage(t2_patient_006)

# Plot the 11th (index 10) slice of the equalized t2-weighted image for patient 0006
plt.imshow(t2_patient_006_arr[10])
plt.show()