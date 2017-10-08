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

imgsPath = 'images/'
img = cv2.imread(imgsPath + 'Boss.bmp')
imgSize = np.shape(img)

gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bImg = cv2.medianBlur(src = gImg, ksize = 51)

threshold, _ = cv2.threshold(src = bImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cannyImg = cv2.Canny(image = bImg, threshold1 = 0.5 * threshold, threshold2 = threshold)


# findContours is a distructive function so the image pased is only a copy
_, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

maxRect = Rect(0, 0, 0, 0)
for contour in contours:
    x, y, w, h = cv2.boundingRect(points = contour)
    currentArea = w * h
    if currentArea > maxRect.getArea():
        maxRect.set(x, y, w, h)

cv2.rectangle(img, (maxRect.x, maxRect.y), (maxRect.x + maxRect.w, maxRect.y + maxRect.h), (0, 0, 255), 1)

cv2.imshow('Page', img)
cv2.waitKey(0)

def getPaperFromImage(img):
    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bImg = cv2.medianBlur(src = gImg, ksize = 51)

    threshold, _ = cv2.threshold(src = bImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyImg = cv2.Canny(image = bImg, threshold1 = 0.5 * threshold, threshold2 = threshold)

    _, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

    maxRect = Rect(0, 0, 0, 0)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(points = contour)
        currentArea = w * h
        if currentArea > maxRect.getArea():
            maxRect.set(x, y, w, h)

    return img[maxRect.y : maxRect.y + maxRect.h, maxRect.x : maxRect.x + maxRect.w]
