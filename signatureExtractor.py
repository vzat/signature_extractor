###############################
#
#   (c) Vlad Zat 2017
#   Student No: C14714071
#   Course: DT228
#   Date: 06-10-2017
#
#	Title: Signature Extractor
#
#   Introduction:
#
#   Describe the algorithm here instead of doing it line by line (10 - 15 lines)
#	Preferably use a systematic approach
#	e.g. step-by-step
#	or pseudocode
#	Give an overview
#	Comment on experiments
#	Use references (Harvard Reference System or IEEE) - no weblinks

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

imgsPath = 'images/'

# TODO Use easygui to open images

signature = cv2.imread(imgsPath + 'Boss.bmp')
# signature = cv2.medianBlur(signature, 3)
# signature = cv2.GaussianBlur(signature, (3, 3), 0)

# TODO Throw error if it's not a valid image

def adaptiveThreshold(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # TODO remove the constrants from here:
    img = cv2.adaptiveThreshold(g, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 21, C = 10)
    # img = cv2.adaptiveThreshold(g, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 21, C = 10)
    return img

mask = adaptiveThreshold(signature)
# k = np.array([[1,4,1], [4,7,4], [1,4,1]], dtype=float)
# blurredSignature = cv2.filter2D(mask,-1,k)

# im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(signature, contours, -1, (0,255,0), 3)

# TODO Clean the mask
# shape = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)

rmask = cv2.bitwise_not(mask)
# shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# rmask = cv2.morphologyEx(rmask, cv2.MORPH_CLOSE, shape)
signature = cv2.bitwise_and(signature, signature, mask=rmask)

# TODO ??? Click on an image to append the signature there

# Reference - http://docs.opencv.org/3.3.0/d4/d13/tutorial_py_filtering.html
k = np.array([[1,4,1], [4,7,4], [1,4,1]], dtype=float)
blurredSignature = cv2.filter2D(signature,-1,k)
imgs = np.hstack((signature, blurredSignature))

sHeight, sWidth, _ = signature.shape
scaledSignature = cv2.resize(signature, (2 * sWidth, 2 * sHeight))

cv2.imshow('Signature', imgs)
key = cv2.waitKey(0)
