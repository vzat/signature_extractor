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

def getPageFromImage(img):
    imgSize = np.shape(img)

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bImg = cv2.medianBlur(src = gImg, ksize = 11)
    bImg = gImg.copy()

    threshold, _ = cv2.threshold(src = bImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyImg = cv2.Canny(image = bImg, threshold1 = 0.5 * threshold, threshold2 = threshold)

    _, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

    maxRect = Rect(0, 0, 0, 0)
    coordinates = []
    bestContour = 0
    index = 0
    for contour in contours:
        epsilon = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.1 * epsilon, True)
        x, y, w, h = cv2.boundingRect(points = contour)
        currentArea = w * h

        if len(corners) == 4 and currentArea > maxRect.getArea():
            maxRect.set(x, y, w, h)
            bestContour = index

        index += 1

    contoursInPage = 0
    for contour in contours:
        x, y, _, _ = cv2.boundingRect(points = contour)
        if (x > maxRect.x and x < maxRect.x + maxRect.w) and (y > maxRect.y and y < maxRect.y + maxRect.h):
                contoursInPage += 1

    maxContours = 5
    if contoursInPage <= maxContours:
        print 'No Page Found'

    print bestContour
    print len(contours)
    print cv2.isContourConvex(contours[bestContour])
    cv2.drawContours(img, contours, bestContour, (0, 0, 255))

    cv2.imshow('Page', img)
    cv2.waitKey(0)

camera = cv2.VideoCapture(0)
(grabbed, signature) = camera.read()

getPageFromImage(signature)

cv2.waitKey(0)
