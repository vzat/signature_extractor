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

signature = cv2.imread(imgsPath + "Boss.bmp")

def adaptiveThreshold(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # TODO remove the constrants from here:
    img = cv2.adaptiveThreshold(g, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 5, C = 15)
    return img

mask = adaptiveThreshold(signature)
# k = np.array([[1,4,1], [4,7,4], [1,4,1]], dtype=float)
# blurredSignature = cv2.filter2D(mask,-1,k)
# cv2.imshow('Signature', blurredSignature)
# key = cv2.waitKey(0)

# TODO Clean the mask

rmask = cv2.bitwise_not(mask)
signature = cv2.bitwise_and(signature, signature, mask=rmask)

# TODO ??? Click on an image to append the signature there

# Reference - http://docs.opencv.org/3.3.0/d4/d13/tutorial_py_filtering.html
# k = np.array([[1,4,1], [4,7,4], [1,4,1]], dtype=float)
# blurredSignature = cv2.filter2D(signature,-1,k)
# imgs = np.hstack((signature, blurredSignature))

sHeight, sWidth, _ = signature.shape
scaledSignature = cv2.resize(signature, (2 * sWidth, 2 * sHeight))

cv2.imshow('Signature', signature)
key = cv2.waitKey(0)
