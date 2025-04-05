# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def _mse(i1: np.ndarray, i2: np.ndarray) -> float:

    i1 = i1.astype(np.float32)
    i2 = i2.astype(np.float32)

    image_shape = np.shape(i1)

    difference = i1 - i2

    squared_difference = difference ** 2
    
    average_sum = np.sum(squared_difference)/(image_shape[0] * image_shape[1])

    return average_sum

def _psnr(i1: np.ndarray, i2: np.ndarray) -> float:

    max_value = 1

    mse_result = _mse(i1, i2)

    max_squared = max_value ** 2

    max_and_mse =  max_squared / mse_result

    psnr = 10 * np.log10(max_and_mse)

    return psnr

def _ssim(i1: np.ndarray, i2: np.ndarray) -> float:

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    
    mean_1 = np.mean(i1)
    mean_2 = np.mean(i2)
    std_1 = np.std(i1)
    std_2 = np.std(i2)
    covar_1_2 = np.cov(i1.flatten(), i2.flatten())[0, 1]
    
    numerator = (2 * mean_1 * mean_2 + c1) * (2 * covar_1_2 + c2)
    denominator = ( mean_1**2 +  mean_2**2 + c1) * (std_1**2 + std_2**2 + c2)
    
    ssim =  numerator / denominator

    return ssim

def _npcc(i1: np.ndarray, i2: np.ndarray) -> float:
    image_1_mean = np.mean(i1)
    image_2_mean = np.mean(i2)

    numerator = np.sum((i1 - image_1_mean) * (i2 - image_2_mean))

    denominator = np.sqrt(np.sum((i1 - image_1_mean) ** 2)) * np.sqrt(np.sum((i2 - image_2_mean) ** 2))

    npcc = numerator / (denominator + 0.00001) # Constant to avoid division by zero

    return npcc

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    result = {
    "mse": _mse(i1, i2),
    "psnr": _psnr(i1, i2),
    "ssim": _ssim(i1, i2),
    "npcc": _npcc(i1, i2)
    }
    return result
