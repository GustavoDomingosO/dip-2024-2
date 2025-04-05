# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

img = np.zeros(shape=[2,3])
for i in range(img.shape[0]):
    img[i][1] = 1

img[0][2] = 2

print(img)

def _translate(img: np.ndarray) -> np.ndarray:

    right_step = 1
    down_step = 1

    result_img = np.zeros(shape=[img.shape[0],img.shape[1]], dtype=img.dtype)

    result_img[right_step:, down_step:] = img[:-right_step, :-down_step]
    return result_img

def _rotate(img: np.ndarray) -> np.ndarray:
    transpose = img.T
    result = np.zeros((transpose.shape[0], transpose.shape[1]), dtype=img.dtype)

    for i in range(transpose.shape[0]):
        for j in range(transpose.shape[1]):
            result[i,j] = transpose[i, transpose.shape[1] - j - 1]

    return result


def _stretch(img: np.ndarray) -> np.ndarray:
    
    height = img.shape[0]
    width =   img.shape[1] 
    
    new_width = int(width * 1.5)
    
    x_new = np.arange(new_width)
    
    x_old = np.clip(x_new / 1.5, 0, width - 1).astype(int)
    
    result = img[:, x_old]

    return result

def _mirror(img: np.ndarray) -> np.ndarray:
    result = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i,j] = img[i, img.shape[1] - j - 1]
    
    return result

import numpy as np

def _distort(img: np.ndarray, k: float = -0.0005) -> np.ndarray:

    height = img.shape[0] 
    width = img.shape[1]
    distorted = np.zeros_like(img)
    
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)
    
    r = np.sqrt(xv**2 + yv**2)
    
    r = np.clip(r, 1e-10, None) 
    
    r_distorted = r * (1 + k * r**2)
    
    x_distorted = xv * (r_distorted / r)
    y_distorted = yv * (r_distorted / r)
    
    x_distorted = ((x_distorted + 1) / 2) * (width - 1)
    y_distorted = ((y_distorted + 1) / 2) * (height - 1)
    
    x_floor = np.floor(x_distorted).astype(int)
    y_floor = np.floor(y_distorted).astype(int)
    x_ceil = np.ceil(x_distorted).astype(int)
    y_ceil = np.ceil(y_distorted).astype(int)
    
    x_floor = np.clip(x_floor, 0, width - 1)
    y_floor = np.clip(y_floor, 0, height - 1)
    x_ceil = np.clip(x_ceil, 0, width - 1)
    y_ceil = np.clip(y_ceil, 0, height - 1)
    
    a = x_distorted - x_floor
    b = y_distorted - y_floor
    
    tl = img[y_floor, x_floor] 
    tr = img[y_floor, x_ceil] 
    bl = img[y_ceil, x_floor]   
    br = img[y_ceil, x_ceil]  
    
    distorted = (
        (1 - a) * (1 - b) * tl +
        a * (1 - b) * tr +
        (1 - a) * b * bl +
        a * b * br
    )
    
    return distorted.astype(img.dtype)


def apply_geometric_transformations(img: np.ndarray) -> dict:
    transformations = {
        "translated": _translate(img), 
        "rotated": _rotate(img),  
        "stretched": _stretch(img), 
        "mirrored": _mirror(img),            
        "barrel_distorted": _distort(img) 
    }
    return transformations