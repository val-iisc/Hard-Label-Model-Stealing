import skimage.draw
from PIL import Image
import numpy as np
import random
import os
import cv2

def _generate_random_colors(num_colors, num_channels, intensity_range):
    if num_channels == 1:
        intensity_range = (intensity_range, )
    elif len(intensity_range) == 1:
        intensity_range = intensity_range * num_channels
    colors = [np.random.randint(r[0], r[1] + 1, size=num_colors)
              for r in intensity_range]
    return np.transpose(colors)

min_size = 5 #5
max_size = 10 #6
#arr = [130000, 150000, 220000]
arr = [50]
count = 0
#arr = [200000, 300000, 500000]
#for num_shapes in [200, 250, 300]:
for num_shapes in [50]:
    len_ = arr[count]
    count+=1
    for image_num in range(len_):
        #image, labels = skimage.draw.random_shapes((32, 32), max_shapes=num_shapes, min_size=12, max_size=22)
        image, labels = skimage.draw.random_shapes((100, 100), max_shapes=num_shapes, min_shapes=num_shapes, min_size=min_size, max_size=max_size, allow_overlap=True)
        num =  _generate_random_colors(1, 3, ((0, 254),))
        #print(num[0])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(3):
                    if image[i,j,k]==255:
                        image[i,j,k] = num[0][k]
        image = cv2.blur(image, (4,4))
        image = cv2.resize(image, (32,32), interpolation = cv2.INTER_NEAREST) 
        im = Image.fromarray(image)
        im.save("./synthetic_data/50k_samples/file_name_" + str(num_shapes) + "_" + str(image_num)+".png")


