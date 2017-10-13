import numpy as np
import cv2
import easygui
#
# def scaleImageDown(img):
#     imgSize = np.shape(img)
#     maxSize = (720, 1280)
#     if imgSize[0] > maxSize[0] or imgSize[1] > maxSize[1]:
#         print 'Warning: Image too big'
#         wRatio = float(float(maxSize[0]) / float(imgSize[0]))
#         hRatio = float(float(maxSize[1]) / float(imgSize[1]))
#         ratio = 1.0
#         if wRatio > hRatio:
#             ratio = wRatio
#         else:
#             ratio = hRatio
#         return cv2.resize(img, (int(ratio * imgSize[1]), int(ratio * imgSize[0])))
#     return img

imgsPath = 'images/'
img = cv2.imread(imgsPath + 'Trump.jpg')

# img = scaleImageDown(img)
gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blockSize = 21
const = 10
mask = cv2.adaptiveThreshold(gImg, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = blockSize, C = const)

_, contours, _ = cv2.findContours(image = mask.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

def getKey(corners):
    return cv2.contourArea(corners)

maxContour = 0
coordinates = []
for contour in contours:
    arcPercentage = 0.1
    epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
    corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
    area = cv2.contourArea(corners)
    if len(corners) == 4:
        coordinates.append(corners)
        if area > maxContour:
            maxContour = area
            bestContour = contour
            x, y, w, h = cv2.boundingRect(points = contour)

sorted(coordinates, key=getKey)

cv2.drawContours(img, coordinates[0], -1, (0, 0, 255), 3)

# cv2.drawContours(img, bestContour, -1, (0, 0, 255), 3)
# img = img[y : y + h, x : x + w]

cv2.imshow('Mask', img)
cv2.waitKey(0)
