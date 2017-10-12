import numpy as np
import cv2
import easygui

def getImageFromFile():
    file = easygui.fileopenbox()
    img = cv2.imread(file)

    if img is None:
        print 'Error: Not a valid image type'
        quit()

    return img

def getImageFromCamera():
    camera = cv2.VideoCapture(0)
    (success, img) = camera.read()

    if not success:
        print 'Error: Cannot capture image'

    return img

def writeImageToFile(img, mask, fileName = 'signature'):
    # # Reference - https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#merge
    b, g, r = cv2.split(img)
    imgWithAlpha = cv2.merge((b, g, r, mask))
    cv2.imwrite(fileName + '.png', imgWithAlpha)

def displayImageToScreen(img, mask):
    imgSize = np.shape(img)
    bg = np.zeros((imgSize[0], imgSize[1], 3), np.uint8)
    bg[:, :] = (255, 255, 255)

    rmask = cv2.bitwise_not(mask)
    bgROI = cv2.bitwise_and(bg, bg, mask = rmask)
    sigROI = cv2.bitwise_and(signature, signature, mask = mask)

    roi = cv2.bitwise_or(bgROI, sigROI)

    cv2.imshow('Signature', roi)
    cv2.waitKey(0)

def scaleImageDown(img):
    imgSize = np.shape(img)
    maxSize = (1920, 1080)
    if imgSize[0] > maxSize[0] or imgSize[1] > maxSize[1]:
        wRatio = float(float(maxSize[0]) / float(imgSize[0]))
        hRatio = float(float(maxSize[1]) / float(imgSize[1]))
        ratio = 1.0
        if wRatio > hRatio:
            ratio = wRatio
        else:
            ratio = hRatio
        return cv2.resize(img, (int(ratio * imgSize[1]), int(ratio * imgSize[0])))

def addPadding(rect, padding, imgSize):
    rect['x'] -= padding
    rect['y'] -= padding
    rect['w'] += 2 * padding
    rect['h'] += 2 * padding
    if rect['x'] < 0:
        rect['x'] = 0
    if rect['y'] < 0:
        rect['y'] = 0
    if rect['x'] + rect['w'] > imgSize[0]:
        rect['w'] = imgSize[0] - rect['x']
    if rect['y'] + rect['h'] > imgSize[1]:
        rect['h'] = imgSize[1] - rect['y']

    return rect

def getPageFromImage(img):
    imgSize = np.shape(img)

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold, _ = cv2.threshold(src = gImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyImg = cv2.Canny(image = gImg, threshold1 = 0.5 * threshold, threshold2 = threshold)

    # findContours() is a distructive function so a copy is passed as a parameter
    _, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print 'Warning: No Page Found'
        return img

    maxRect = {
        'x': 0,
        'y': 0,
        'w': 0,
        'h': 0
    }
    coordinates = []
    for contour in contours:
        arcPercentage = 0.1
        epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
        corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
        x, y, w, h = cv2.boundingRect(points = contour)
        currentArea = w * h

        if len(corners) == 4:
            coordinates.append((x, y))
            if currentArea > maxRect['w'] * maxRect['h']:
                maxRect['x'] = x
                maxRect['y'] = y
                maxRect['w'] = w
                maxRect['h'] = h

    if maxRect['w'] <= 1 or maxRect['h'] <= 1:
        print 'Warning: No Page Found'
        return img

    contoursInPage = 0
    for coordinate in coordinates:
        x = coordinate[0]
        y = coordinate[1]
        if (x > maxRect['x'] and x < maxRect['x'] + maxRect['w']) and (y > maxRect['y'] and y < maxRect['y'] + maxRect['h']):
            contoursInPage += 1

    maxContours = 5
    if contoursInPage <= maxContours:
        print 'Warning: No Page Found'
        return img

    return img[maxRect['y'] : maxRect['y'] + maxRect['h'], maxRect['x'] : maxRect['x'] + maxRect['w']]

def getSignatureFromPage(img):
    imgSize = np.shape(img)

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold, _ = cv2.threshold(src = gImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyImg = cv2.Canny(image = gImg, threshold1 = 0.5 * threshold, threshold2 = threshold)

    # The kernel is wide as most signature are wide
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (30, 1))
    cannyImg = cv2.morphologyEx(src = cannyImg, op = cv2.MORPH_CLOSE, kernel = kernel)

    # findContours() is a distructive function so a copy is passed as a parameter
    _, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print 'Warning: No Signature Found'
        return img

    maxRect = {
        'x': 0,
        'y': 0,
        'w': 0,
        'h': 0
    }
    maxCorners = 0
    for contour in contours:
        arcPercentage = 0.01
        epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
        corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
        x, y, w, h = cv2.boundingRect(points = contour)

        if len(corners) > maxCorners:
            maxCorners = len(corners)
            maxRect['x'] = x
            maxRect['y'] = y
            maxRect['w'] = w
            maxRect['h'] = h

    if maxRect['w'] <= 1 or maxRect['h'] <= 1:
        print 'Warning: No Signature Found'
        return img

    # Add padding so the signature is more visible
    maxRect = addPadding(rect = maxRect, padding = 10, imgSize = imgSize)

    return img[maxRect['y'] : maxRect['y'] + maxRect['h'], maxRect['x'] : maxRect['x'] + maxRect['w']]

def getSignature(img):
    imgSize = np.shape(img)

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding requires the blocksize to be even and bigger than 1
    blockSize = 1 / 8 * imgSize[0] / 2 * 2 + 1
    if blockSize <= 1:
        blockSize = imgSize[0] / 2 * 2 + 1
    const = 10

    mask = cv2.adaptiveThreshold(gImg, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = blockSize, C = const)
    rmask = cv2.bitwise_not(mask)

    return (cv2.bitwise_and(img, img, mask=rmask), rmask)

# First Prompt
title = 'Image Selection'
message = 'Choose a method of getting the picture'
buttons = ['File', 'Camera']
selection = easygui.indexbox(msg = message, title = title, choices = buttons)

if selection == 0:
    img = getImageFromFile()
elif selection == 1:
    img = getImageFromCamera()
else:
    quit()

# Make the image smaller if the image is too big as the signature extraction does not work properly with large images
img = scaleImageDown(img)

# Extract Signature
page = getPageFromImage(img = img)
signatureBlock = getSignatureFromPage(img = page)
(signature, mask) = getSignature(img = signatureBlock)

# Second Prompt
title = 'Display or Export'
message = 'Choose a method of showing the signature'
buttons = ['Display on Screen', 'Export to File']
selection = easygui.indexbox(msg = message, title = title, choices = buttons)

if selection == 0:
    displayImageToScreen(signature, mask)
elif selection == 1:
    writeImageToFile(signature, mask)
else:
    quit()

# buttonbox?
# Buttons:
# select file
# capture image from camera
# export signature to file
# display signature
# maybe extra features (checkboxes?) to improved the detection

# cv2.imshow('Test', img)
# cv2.waitKey(0)
