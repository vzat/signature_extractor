# Get Signature from Page:
# convert image to grayscale
# get edges of the signature
# close the result
# find contours
# find the contour with the biggest bounding rectangle area
# add padding to the bounding rectangle
# generate a new image that only contains the largest bounding rectangle

# Extra Notes:
# Filtering is not necessary because writting doesn't have a texture to impede edge detection
# Filtering in this case only makes the text harder to read


import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

imgsPath = 'images/'

img = cv2.imread(imgsPath + 'Boss.bmp')
imgSize = np.shape(img)

gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Filtering is not necessary because writting doesn't have a texture to impede edge detection
# Filtering in this case only makes the text harder to read

# The necessary value for edge detection using Canny can be approximated using Otsu's Algorithm
# Reference - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.5899&rep=rep1&type=pdf
threshold, _ = cv2.threshold(gImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cannyImg = cv2.Canny(gImg, 0.5 * threshold, threshold)

# Close the image to fill blank spots so blocks of text that are close together (like the signature)
# are easier to detect
# The shape of the rect structuring elemnt should have a large width and short height because
# signature are usually like this
shape = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
cannyImg = cv2.morphologyEx(cannyImg, cv2.MORPH_CLOSE, shape)

# findContours is a distructive function so the image pased is only a copy
_, contours, _ = cv2.findContours(cannyImg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


class Rect:
    def __init__(self, x = 0, y = 0, w = 0, h = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def getArea(self):
        return self.w * self.h
    def set(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def addPadding(self, imgSize, padding):
        self.x -= padding
        self.y -= padding
        self.w += 2 * padding
        self.h += 2 * padding
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x + self.w > imgSize[0]:
            self.w = imgSize[0] - self.x
        if self.y + self.h > imgSize[1]:
            self.h = imgSize[1] - self.y


maxRect = Rect(0, 0, 0, 0)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    currentArea = w * h
    if currentArea > maxRect.getArea():
        maxRect.set(x, y, w, h)

# Increase the bounding box to get a better area of the signature
maxRect.addPadding(imgSize, 10)
cv2.rectangle(img, (maxRect.x, maxRect.y), (maxRect.x + maxRect.w, maxRect.y + maxRect.h), (0, 0, 255), 1)


cv2.imshow('Signature', img)
key = cv2.waitKey(0)

# TODO Put this into a function that return a new image that just contains the signature





def getSignatureFromPage(img):
    imgSize = np.shape(img)

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The values for edge detection can be approximated using Otsu's Algorithm
    # Reference - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.5899&rep=rep1&type=pdf
    threshold, _ = cv2.threshold(src = gImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyImg = cv2.Canny(image = gImg, threshold1 = 0.5 * threshold, threshold2 = threshold)

    # Close the image to fill blank spots so blocks of text that are close together (like the signature) are easier to detect
    # Signature usually are wider and shorter so the strcturing elements used for closing will have this ratio
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (30, 1))
    cannyImg = cv2.morphologyEx(src = cannyImg, op = cv2.MORPH_CLOSE, kernel = kernel)

    # findContours is a distructive function so the image pased is only a copy
    _, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

    maxRect = Rect(0, 0, 0, 0)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(points = contour)
        currentArea = w * h
        if currentArea > maxRect.getArea():
            maxRect.set(x, y, w, h)

    # Increase the bounding box to get a better view of the signature
    maxRect.addPadding(imgSize = imgSize, padding = 10)

    return img[maxRect.y : maxRect.y + maxRect.h, maxRect.x : maxRect.x + maxRect.w]
