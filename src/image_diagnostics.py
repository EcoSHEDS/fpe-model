from typing import Union

import cv2
import numpy as np
from PIL import Image


def standardize_image_array(
    image: Union[Image.Image, np.ndarray], is_bgr: bool = False
) -> np.ndarray:
    """
    Converts a PIL Image or OpenCV image to a standardized numpy array in RGB format.

    Args:
        image (Union[Image.Image, np.ndarray]): Input image, either a PIL Image or an OpenCV image (numpy array).
        is_bgr (bool, optional): If True, converts BGR image to RGB. Assumes RGB if False. Default is False.

    Returns:
        np.ndarray: Standardized image as a numpy array (height, width, channels), with RGB channel order.
    """
    # Convert the image to a numpy array if it's a PIL Image object
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert the image from BGR to RGB if it's in BGR format
    if image.shape[-1] == 3 and is_bgr:
        image = image[:, :, ::-1]

    return image


def is_grayscale(image: np.ndarray) -> bool:
    """
    Checks if the image is grayscale.

    For a single-channel image, it is considered grayscale.
    For a three-channel image, it is considered grayscale if all channels are identical.

    Args:
        image (np.ndarray): Input image as a numpy array with RGB channel order.

    Returns:
        bool: True if the image is grayscale, False otherwise.
    """
    if len(image.shape) == 2:
        return True
    elif len(image.shape) == 3:
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        if np.array_equal(r, g) and np.array_equal(r, b):
            return True
    return False


def mean_absolute_lightness(image: np.ndarray) -> float:
    """
    Compute the mean lightness of an image.

    This function can be useful for identifying images with extremely high
    or low mean lightness, which might be overexposed or underexposed,
    respectively. These images might not contain enough detail for further
    processing or analysis, and could be candidates for lightness correction
    or exclusion from the dataset.

    Note that while lightness can be adjusted to some extent to correct for
    overexposure or underexposure, severe cases can result in loss of detail
    that cannot be fully recovered.

    Parameters:
    image (np.ndarray): The input image in RGB format.

    Returns:
    float: The mean lightness of the image, normalized to the range [0, 1].
    """
    assert image.dtype == np.uint8, "Image must be of type np.uint8"
    assert image.ndim == 3, "Image must have 3 dimensions"
    assert image.shape[2] == 3, "Image must have 3 channels"

    # convert RGB to LAB
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2LAB))
    # normalize
    L = L / 255.0
    # return mean
    return np.mean(L)


def dark_channel_prior(
    image: np.ndarray, omega: float = 0.95, window_size: int = 15, max_size: int = 500
) -> np.ndarray:
    """
    Computes the transmission map and the atmospheric light of an image using the dark channel prior.

    See 'Single Image Haze Removal Using Dark Channel Prior' by Kaiming He et al. for more details.

    The transmission map represents the portion of the light that is not scattered and reaches the camera.
    It is useful for estimating the amount of haze in an image because a lower transmission value indicates
    more light being scattered away from the line of sight, which corresponds to more haze.

    The atmospheric light is the intensity of the ambient light. It is useful for estimating the amount of
    haze in an image because a higher atmospheric light value indicates more light being scattered into the
    line of sight, which corresponds to more haze.

    Note: The order of the color channels (RGB or BGR) does not affect the result.

    Args:
        image (np.ndarray): Input image as a numpy array with RGB channel order.
        omega (float): Amount of atmospheric light to keep.
        window_size (int): Size of the window used to compute the minimum filter.

    Returns:
        np.ndarray: The transmission map of the image.
        float: The atmospheric light of the image.
    """
    assert isinstance(image, np.ndarray), "Image must be a numpy array."
    assert (
        len(image.shape) == 3
    ), "Image must have three dimensions (height, width, and color channels)."
    assert image.shape[2] >= 3, "Image must have at least three color channels."

    # Get the height and width of the image
    height, width = image.shape[:2]

    # If the image is larger than the desired size, reduce its resolution
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    # Normalize the image to the range [0, 1]
    normalized_image = image.astype(np.float32) / 255

    # Compute the dark channel
    dark_channel = cv2.min(
        cv2.min(normalized_image[:, :, 0], normalized_image[:, :, 1]),
        normalized_image[:, :, 2],
    )
    # Erode the dark channel to remove small bright regions and enhance large dark regions
    dark_channel = cv2.erode(dark_channel, np.ones((window_size, window_size)))

    # Estimate the atmospheric light
    atmospheric_light = np.max(normalized_image)

    # Compute the transmission map
    transmission = 1 - omega * dark_channel / atmospheric_light

    return transmission, atmospheric_light


def haze_score(
    image: np.ndarray,
    omega: float = 0.95,
    window_size: int = 15,
) -> float:
    """
    Computes the haze score of an image using the dark channel prior.

    The haze score is based on the mean transmission.

    The mean transmission represents the average portion of the light that is not scattered and
    reaches the camera. A lower mean transmission indicates more light being scattered away from
    the line of sight, which corresponds to more haze.

    The atmospheric light represents the intensity of the ambient light and can affect the
    appearance of haze in an image. However, since it depends on lighting conditions, it can vary
    independently of the amount of haze and is therefore not included in this computation of the
    haze score.

    Args:
        image (np.ndarray): Input image as a numpy array with RGB channel order.
        omega (float): Amount of atmospheric light to keep.
        window_size (int): Size of the window used to compute the minimum filter.

    Returns:
        float: The haze score of the image.
    """
    transmission, _ = dark_channel_prior(image, omega, window_size)
    haze_score = 1 - np.mean(transmission)
    return haze_score


def variance_of_laplacian(image: np.ndarray, ksize: int = 3) -> float:
    """
    Compute the variance of the Laplacian of an image.

    The variance of the Laplacian is a measure of the amount of edges or
    texture in the image. A low variance of the Laplacian indicates that the
    image has few edges or little texture, which could be due to the image being
    blurry, overly smooth, or obscured by noise or haze.

    Args:
        image (np.ndarray): The input image.
        ksize (int): The size of the Laplacian kernel. Default is 3.

    Returns:
        float: The variance of the Laplacian of the image.
    """
    image_blur = cv2.GaussianBlur(image, (3, 3), 0)
    image_laplacian = cv2.Laplacian(image_blur, cv2.CV_64F, ksize=ksize)
    return image_laplacian.var()
