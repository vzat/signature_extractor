import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui


class Rect:
    def __init__(self, x = 0, y = 0, w = 0, h = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = 0

    def setArea(self, area):
        self.area = area
    def getArea(self):
        return self.area
    def set(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h
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



imgsPath = 'images/'
img = cv2.imread(imgsPath + 'Boss.bmp')
gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


imgSize = np.shape(img)
#
# kSize = 11
#
# maxMean = 0
# noBlocks = 0
# sumBlocks = 0
# for x in range(0, imgSize[0], kSize):
#     for y in range(0, imgSize[1], kSize):
#         no = 0
#         sum = 0
#         for lX in range(x - kSize / 2, x + kSize / 2):
#             if lX >= 0 and lX < imgSize[0]:
#                 for lY in range(y - kSize / 2, y + kSize / 2):
#                     if lY >= 0 and lY < imgSize[1]:
#                         no += 1
#                         sum += gImg[lX, lY]
#         mean = sum / no
#         noBlocks += 1
#         sumBlocks += mean
# maxMean = sumBlocks / noBlocks
#
# print maxMean
#
# T = maxMean
# T, mask = cv2.threshold(src = gImg, thresh = T, maxval = 255, type = cv2.THRESH_BINARY)
# rmask = cv2.bitwise_not(mask)
# roi = cv2.bitwise_and(gImg, gImg, mask = mask)
# cv2.imshow('Paper', roi)
#
# cv2.waitKey(0)

# threshold, mask = cv2.threshold(src = gImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# roi = cv2.bitwise_and(gImg, gImg, mask = mask)
#
#
#
# cv2.imshow('Paper', roi)
# cv2.waitKey(0)
