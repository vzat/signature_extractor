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
img = cv2.imread(imgsPath + 'sig3.jpg')
gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# bImg = gImg.copy()
N = 9
# bImg = cv2.blur(src = gImg, ksize = (N, N))
bImg = cv2.medianBlur(src = gImg, ksize = N)

# threshold, _ = cv2.threshold(src = bImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cannyImg = cv2.Canny(image = bImg, threshold1 = 0.5 * threshold, threshold2 = threshold)
#
# threshold2, _ = cv2.threshold(src = mbImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cannyImg2 = cv2.Canny(image = mbImg, threshold1 = 0.5 * threshold, threshold2 = threshold)
#
# # canny = np.hstack((cannyImg, cannyImg2))
# # cv2.imshow('Canny', canny)
# # cv2.waitKey(0)
#
# def displayRects(img, blurType = 'STD_BLUR', kSize = 11):
#     gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     bImg = gImg.copy()
#
#     if blurType == 'STD_BLUR':
#         bImg = cv2.blur(src = gImg, ksize = (kSize, kSize))
#     elif blurType == 'MEDIAN_BLUR':
#         bImg = cv2.medianBlur(src = gImg, ksize = kSize)
#
#     threshold, _ = cv2.threshold(src = bImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     cannyImg = cv2.Canny(image = bImg, threshold1 = 0.5 * threshold, threshold2 = threshold)
#
#     _, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
#
#     maxRect = Rect(0, 0, 0, 0)
#     for contour in contours:
#         epsilon = cv2.arcLength(contour, True)
#         corners = cv2.approxPolyDP(contour, 0.1 * epsilon, True)
#         x, y, w, h = cv2.boundingRect(points = contour)
#         currentArea = w * h
#
#         if len(corners) == 4:
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
#
#     cv2.imshow('Rects', img)
#
#
# displayRects(img, 'STD_BLUR', 11)
# cv2.waitKey(0)
# displayRects(img, 'MEDIAN_BLUR', 11)
# cv2.waitKey(0)

threshold, _ = cv2.threshold(src = bImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cannyImg = cv2.Canny(image = bImg, threshold1 = 0.5 * threshold, threshold2 = threshold)
# cv2.imshow('Canny', cannyImg)
# cv2.waitKey(0)

_, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

maxRect = Rect(0, 0, 0, 0)
for contour in contours:
    epsilon = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, 0.1 * epsilon, True)
    x, y, w, h = cv2.boundingRect(points = contour)
    currentArea = w * h
    # currentArea = cv2.contourArea(contour)

    # check if length of approx is 4
    # if len(corners) == 4 and
    if currentArea > maxRect.getArea():
        maxRect.set(x, y, w, h)
        # maxRect.setArea(currentArea)

    # if len(corners) == 4:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

cv2.rectangle(img, (maxRect.x, maxRect.y), (maxRect.x + maxRect.w, maxRect.y + maxRect.h), (0, 0, 255), 1)

cv2.imshow('Image', img)
cv2.waitKey(0)
