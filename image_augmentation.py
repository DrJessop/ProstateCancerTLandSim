import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import pprint as pp


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
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively
    :param image: An sitk MRI image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param show: Boolean, whether or not the user wants to see the result of the rotation
    :return: The rotated image
    """
    theta_x = np.deg2rad(theta_x)
    theta_y = np.deg2rad(theta_y)
    theta_z = np.deg2rad(theta_z)
    euler_transform = sitk.Euler3DTransform(get_center(image), theta_x, theta_y, theta_z, (0, 0, 0))
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)
    euler_transform.SetRotation(theta_x, theta_y, theta_z)
    resampled_image = resample(image, euler_transform)
    if show:
        plt.imshow(sitk.GetArrayFromImage(resampled_image)[0])
        plt.show()
    return resampled_image


def translation3d(image, trans_x, trans_y, trans_z, show=False):
    """
    This function rotates an image across each of the x,y, z axes by trans_x, trans_y, and trans_x mm
    respectively
    :param image: An sitk MRI image
    :param trans_x: The amount of mm the user wants the image translated by x mm
    :param trans_y: The amount of mm the user wants the image translated by y mm
    :param trans_z: The amount of mm the user wants the image translated by z mm
    :param show: Boolean, whether or not the user wants to see the result of the translation
    :return: The translated image
    """
    euler_transform = sitk.Euler3DTransform()
    image_center = get_center(image)
    euler_transform.SetTranslation((trans_x, trans_y, trans_z))
    resampled_image = resample(image, euler_transform)
    if show:
        plt.imshow(sitk.GetArrayFromImage(resampled_image)[0])
        plt.show()
    return resampled_image


if __name__ == "__main__":
    img = sitk.ReadImage("/home/andrewg/PycharmProjects/assignments/resampled/t2/ProstateX-0000.nrrd")
    img_arr = sitk.GetArrayFromImage(img)[0]
    plt.imshow(img_arr)
    plt.show()

    input("Press enter to continue...")

    # rotation3d(img, 0, 0, 45, show=True)
    translation3d(img, 50, 50, 0, show=True)

    pp.pprint(sorted(os.listdir("/home/andrewg/PycharmProjects/assignments/resampled/t2")))
