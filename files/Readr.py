# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:28:37 2021

@author: Ranji Raj
"""
import os
import cv2
from cv2 import imread, resize
import numpy as np

input = cv2.imread("C:\\Users\\User\\Desktop\\Easy_CM.png")

cv2.imshow("Confusion Matrix", input)

cv2.waitKey()

cv2.destroyAllWindows()
