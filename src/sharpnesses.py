import numpy as np

from skimage.feature import canny
from skimage.filters import sobel, laplace


# A function for computing the gradient sharpness of an image
# It is based on the averaging the gradient norm
def gradient_sharpness(image):
    array = np.asarray(image)
    gy, gx = np.gradient(array)
    norm = np.sqrt(gx ** 2 + gy ** 2)

    return np.average(norm)


# A function for computing the sharpness using the Sobel edge detector
def sobel_sharpness(image):
    return np.sum(np.array(sobel(image)))


# A function for computing the sharpness using the Canny edge detector
def canny_sharpness(image):
    return np.sum(canny(image))


# A function for computing the sharpness using the Laplacian edge detector
def laplacian_sharpness(image):
    return np.sum(laplace(image))
