import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def resample(image, transform):
    """
    This function resamples (updates) an image using a specified transform
    :param image: The sitk image we are trying to transform
    :param transform: An sitk transform (ex. resizing, rotation, etc.
    :return: The transformed sitk image
    """
    reference_image = image
    interpolator = sitk.sitkBSpline
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def get_center(img):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                              int(np.ceil(height/2)),
                                              int(np.ceil(depth/2))))


def rotation3d(image, theta_x, theta_y, theta_z, show=False):
    # theta_x, theta_y, theta_z = deg_to_rad()
    theta_x = np.pi * theta_x / 180
    theta_y = np.pi * theta_y / 180
    theta_z = np.pi * theta_z / 180
    euler_transform = sitk.Euler3DTransform()
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)
    # print(euler_transform.GetCenter())
    euler_transform.SetRotation(theta_x, theta_y, theta_z)
    # euler_transform.SetTranslation((0, 2.5, 0))
    resampled_image = resample(image, euler_transform)
    if show:
        plt.imshow(sitk.GetArrayFromImage(resampled_image)[0])
        plt.show()
    return resampled_image
