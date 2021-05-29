# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 22:41:14 2021

@author: Ranji Raj
"""

import os
import cv2 
from cv2 import imread, resize
import numpy as np
from matplotlib import pyplot as plt


def imreads(path):
    
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    for image_path in images_path:
        img = cv2.imread(image_path)
        images.append(img)
    return images


imgs_path = "C:\\Users\\User\\Downloads\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\JPEGImages/"  

dictionary_size = 512
# Loading images
imgs_data = []

imgs = imreads(imgs_path)
for i in range(len(imgs)):
    # create a numpy to hold the histogram for each image
    imgs_data.insert(i, np.zeros((dictionary_size, 1)))


def get_descriptors(img, detector):
    # returns descriptors of an image
    return detector.detectAndCompute(img, None)[1]


# Extracting descriptors
detector = cv2.AKAZE_create()

desc = np.array([])
# desc_src_img is a list which says which image a descriptor belongs to
desc_src_img = []
for i in range(len(imgs)):
    img = imgs[i]
    descriptors = get_descriptors(img, detector)
    if len(desc) == 0:
        desc = np.array(descriptors)
    else:
        desc = np.vstack((desc, descriptors))
    # Keep track of which image a descriptor belongs to
    for j in range(len(descriptors)):
        desc_src_img.append(i)
# important, cv2.kmeans only accepts type32 descriptors
desc = np.float32(desc)

# Clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
flags = cv2.KMEANS_PP_CENTERS
compactness, labels, dictionary = cv2.kmeans(desc, dictionary_size, None, criteria, 1, flags)

# Getting histograms from labels
size = labels.shape[0] * labels.shape[1]
for i in range(size):
    label = labels[i]
    # Get this descriptors image id
    img_id = desc_src_img[i]
    # imgs_data is a list of the same size as the number of images
    data = imgs_data[img_id]
    # data is a numpy array of size (dictionary_size, 1) filled with zeros
    data[label] += 1

print( "Histogram from labels: ")
print( imgs_data[0].ravel())
ax = plt.subplot(311)
ax.set_title("Histogram from labels")
ax.set_xlabel("Visual words")
ax.set_ylabel("Frequency")
ax.plot(imgs_data[0].ravel())
