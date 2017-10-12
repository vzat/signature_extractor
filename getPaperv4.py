import numpy as np
import cv2
import easygui

def getImageFromCamera():
    camera = cv2.VideoCapture(0)
    (success, img) = camera.read()

    if not success:
        print 'Error: Cannot capture image'

    return img

imgsPath = 'images/'
img = cv2.imread(imgsPath + 'Boss.bmp')
# img = getImageFromCamera()
# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# img = cv2.filter2D(img, -1, kernel)
# ratio = 0.2
# imgSize = np.shape(img)
# img = cv2.resize(img, (int(ratio * imgSize[1]), int(ratio * imgSize[0])))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s = hsv[:, :, 1]
gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gImg = cv2.blur(gImg, (7, 7))
gImg = cv2.medianBlur(gImg, 7, 7)
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y = yuv[:, :, 0]

# threshold, cannyImg = cv2.threshold(src = gImg, thresh = 200, maxval = 255, type = cv2.THRESH_BINARY)
# cv2.imshow('test', cannyImg)
# cv2.waitKey(0)
threshold, _ = cv2.threshold(src = gImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cannyImg = cv2.Canny(image = gImg, threshold1 = 0.5 * threshold, threshold2 = threshold)

# kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = (5, 5))
# cannyImg = cv2.morphologyEx(src = cannyImg, op = cv2.MORPH_CLOSE, kernel = kernel)

cv2.imshow('Canny', cannyImg)
cv2.waitKey(0)

_, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)

maxArea = 0
for contour in contours:
    hull = cv2.convexHull(contour)
    # hull = contour
    epsilon = cv2.arcLength(hull, True)
    sContour = cv2.approxPolyDP(hull, 0.1 * epsilon, True)
    x, y, w, h = cv2.boundingRect(points = sContour)

    area = cv2.contourArea(sContour)
    if area > maxArea:
        maxArea = area
        bestContour = contour

    # x, y, w, h = cv2.boundingRect(points = sContour)
    # if w * h > 2000:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

cv2.drawContours(img, bestContour, -1, (0, 0, 255), 5)
# x, y, w, h = cv2.boundingRect(points = bestContour)
# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)


# th = 50
# th = np.mean(s) + np.std(s)
# th, paper = cv2.threshold(src = s, thresh = th, maxval = 255, type = cv2.THRESH_BINARY)

cv2.imshow('Paper', img)
cv2.waitKey(0)
