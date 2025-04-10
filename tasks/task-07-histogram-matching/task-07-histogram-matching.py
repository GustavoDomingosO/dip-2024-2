# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt

import numpy as np

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:

    matched_channels = []
    
    for channel in range(3):
        source_channel = source_img[:, :, channel]
        reference_channel = reference_img[:, :, channel]
        
        src_hist, _ = np.histogram(source_channel.flatten(), 256, [0,256])
        src_cdf = src_hist.cumsum()
        src_cdf_normalized = (src_cdf - src_cdf.min()) * 255 / (src_cdf.max() - src_cdf.min())
        
        ref_hist, _ = np.histogram(reference_channel.flatten(), 256, [0,256])
        ref_cdf = ref_hist.cumsum()
        ref_cdf_normalized = (ref_cdf - ref_cdf.min()) * 255 / (ref_cdf.max() - ref_cdf.min())
        
        lookup_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            idx = np.argmin(np.abs(src_cdf_normalized[i] - ref_cdf_normalized))
            lookup_table[i] = idx
            
        matched_channel = lookup_table[source_channel]
        matched_channels.append(matched_channel)
    
    matched_img = np.stack(matched_channels, axis=2).astype(np.uint8)
    return matched_img

reference_img = cv.imread('tasks/task-07-histogram-matching/reference.jpg')
source_img = cv.imread('tasks/task-07-histogram-matching/source.jpg')

source_img = cv.cvtColor(source_img, cv.COLOR_BGR2RGB)
reference_img = cv.cvtColor(reference_img, cv.COLOR_BGR2RGB)

matched_img = match_histograms_rgb(source_img, reference_img)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(source_img)
plt.title('Source Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(reference_img)
plt.title('Reference Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(matched_img)
plt.title('Matched Image')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
colors = ('red', 'green', 'blue')
channel_names = ('Red', 'Green', 'Blue')

for i, color in enumerate(colors):

    plt.subplot(3, 3, i + 1)
    plt.hist(source_img[:, :, i].flatten(), bins=256, color=color, alpha=0.7, range=(0, 256))
    plt.title(f'Source {channel_names[i]} Histogram')
    plt.xlim([0, 256])

    plt.subplot(3, 3, i + 4)
    plt.hist(reference_img[:, :, i].flatten(), bins=256, color=color, alpha=0.7, range=(0, 256))
    plt.title(f'Reference {channel_names[i]} Histogram')
    plt.xlim([0, 256])

    plt.subplot(3, 3, i + 7)
    plt.hist(matched_img[:, :, i].flatten(), bins=256, color=color, alpha=0.7, range=(0, 256))
    plt.title(f'Matched {channel_names[i]} Histogram')
    plt.xlim([0, 256])

plt.tight_layout()
plt.show()