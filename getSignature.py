import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

imgsPath = 'images/'

img = cv2.imread(imgsPath + 'Boss.bmp')

gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Bilateral Filtering
# Reference - https://people.csail.mit.edu/sparis/publi/2009/fntcgv/Paris_09_Bilateral_filtering.pdf
gImg = cv2.bilateralFilter(gImg, 11, 17, 17)

cv2.imshow('Signature', gImg)
key = cv2.waitKey(0)


# Step 1
# Filter Image for better edge detection - Look for ways to determine the values (sigma)

# Step 2
# Use Canny to get the edges - Look for ways to determine the values (someone called He?)

# Step 3
# Find the contours and get the largest(?) rect in the image

# Step 4
# Use adaptive thresholding to get the signature

# Step 5
# Use a mask to get the actual signature with the correct colour
