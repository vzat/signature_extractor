###############################
#
#   (c) Vlad Zat 2017
#   Student No: C14714071
#   Course: DT228
#   Date: 13-10-2017
#
#	Title: Signature Extractor
#
#   Introduction:
#   Extract the signature from an image by firstly extracting the page in it (if any)
#   then extracting a signature block, and in the end use thresholding to extract only the
#   signature in the correct colour. Allow the user to import an image from file or capture
#   from the camera. The result can be viewed on the screen or exported to file.
#
#   Structure:
#   1. Extract the Page
#       a. Convert the image to Grayscale
#           * Using only one channel is necessary for both edge detection and segmentation
#       b. Find the edges of the image using Canny
#           * Edge detection is used to be able to find the main objects in the image
#           * The range for Canny is approximated to be from half the treshhold to
#             the value of the treshhold. This threshold is calculated using Otsu's Algorithm.
#             This provides consident results and is recommended by Mei Fang (et al.) in
#             "The Study on An Application of Otsu Method in Canny Operator" [1]
#       c. Getting the contours of the objects in the image
#       d. Finding the biggest contour with 4 edges
#           * The perimeter of the contour is calculated and then used to approximate a
#             polygon around it. To decrese the ammound of edges detected, the permited
#             is multiplied with 0.1 as recommended in the OpenCV Documentation [2]
#       e. Detecting if the biggest contour has any contours in it
#           * If the biggest contour does not have any other contours in it (such as words
#             or the signature) then it's a false alarm and there is no complete page in the image
#             so the whole image is used in the next step
#   2. Extracting the Signature
#       a. Convert the image to Grayscale
#       b. Find the edges of the image using Canny
#       c. Getting the contours of the objects in the image
#       d. Closing the image so the signatures is more filled and pronounced
#       e. Finding the contour with the most number of edges
#           * Opposed to the method from the first step, here the permimeter of the
#             signature is multipled with 0.01 to give it a better shape of the signature [2]
#           * Signatures have more edges than normal text in a page so the contour with
#             the maximum ammount of edges should be the signature
#       f. Pad the image so the signature can be viewed more easily
#   3. Remove the background from the signature
#       a. Convert the image to Grayscale
#       b. Use adaptive thresholding to extract only the signature
#           * A block of 1/8 of the width of the image is used as it provides
#             consistent results
#           * If the image is too small then the width of the image is used instead
#
#   Experiments:
#       * Blur the image using Median Blur and Bilateral Blur to maintain the edges
#         but smooth the objects. It requires hardcoded values for the kernel size used
#         and it doesn't provide good enough results
#       * Use thresholding to extract the page. While it does provide similar results to
#         the edge detection method, it requires hard coded values
#       * Try to use Otsu's Algorithm for the final step. It does not provide good results,
#         even for small parts of the image
#       * Use Hough Line to get the lines in the image and extracting zones where they are
#         intersecting. It does not detect lines in the images
#       * Use a feature detection algorithm such as AKAZE to get the the features of the image.
#         It doesn't provide better results than using Canny
#       * Use different colour spaces to better extract the paper. The luminance or saturation
#         provide similar results to grayscale
#       * Make the image smaller if the image is too big as the signature extraction does not
#         work properly with large images. It causes problems for some images.
#
#   References:
#       [1] M.Fang, GX.Yue1, QC.Yu, 'The Study on An Application of Otsu Method in Canny Operator',
#           International Symposium on Information Processing, Huangshan, P. R. China,
#           August 21-23, 2009, pp. 109-112
#       [2] OpenCV, 'Contour Approximation', 2015. [Online].
#           Available: http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
#           [Accessed: 2017-10-05]

import numpy as np
import cv2
import easygui

def getImageFromFile():
    file = easygui.fileopenbox()
    img = cv2.imread(file)

    if img is None:
        errorPrompt('Not a valid image type')

    return img

def getImageFromCamera():
    camera = cv2.VideoCapture(0)
    (success, img) = camera.read()

    if not success:
        errorPrompt('Cannot capture image')

    return img

def writeImageToFile(img, mask):
    # The mask of the signature can be used as the alpha channel of the image
    b, g, r = cv2.split(img)
    imgWithAlpha = cv2.merge((b, g, r, mask))

    file = easygui.filesavebox()
    fileName = file + '.png'

    if fileName is None:
        errorPrompt('No Name Selected')

    cv2.imwrite(fileName, imgWithAlpha)

def displayImageToScreen(img, mask):
    imgSize = np.shape(img)
    bg = np.zeros((imgSize[0], imgSize[1], 3), np.uint8)
    bg[:, :] = (255, 255, 255)

    # Add a white background to the signature
    rmask = cv2.bitwise_not(mask)
    bgROI = cv2.bitwise_and(bg, bg, mask = rmask)
    sigROI = cv2.bitwise_and(signature, signature, mask = mask)

    roi = cv2.bitwise_or(bgROI, sigROI)

    cv2.imshow('Signature', roi)
    cv2.waitKey(0)

def errorPrompt(errorMsg):
    print 'Error: ' + errorMsg
    easygui.msgbox(msg = errorMsg, title = 'ERROR', ok_button = 'OK')
    quit()

def scaleImageDown(img):
    imgSize = np.shape(img)
    maxSize = (1920, 1080)
    if imgSize[0] > maxSize[0] or imgSize[1] > maxSize[1]:
        print 'Warning: Image too big'
        wRatio = float(float(maxSize[0]) / float(imgSize[0]))
        hRatio = float(float(maxSize[1]) / float(imgSize[1]))
        ratio = 1.0
        if wRatio > hRatio:
            ratio = wRatio
        else:
            ratio = hRatio
        return cv2.resize(img, (int(ratio * imgSize[1]), int(ratio * imgSize[0])))
    return img

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
        # Perimeter accuracy
        arcPercentage = 0.1

        # Contour Perimeter
        epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
        corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
        x, y, w, h = cv2.boundingRect(points = corners)
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

    if contoursInPage <= 0:
        print 'Warning: No Page Found'
        return img

    return img[maxRect['y'] : maxRect['y'] + maxRect['h'], maxRect['x'] : maxRect['x'] + maxRect['w']]

def getSignatureFromPage(img):
    imgSize = np.shape(img)

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold, _ = cv2.threshold(src = gImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyImg = cv2.Canny(image = gImg, threshold1 = 0.5 * threshold, threshold2 = threshold)

    # The kernel is wide as most signatures are wide
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
        # Perimeter accuracy
        arcPercentage = 0.01

        # Contour perimeter
        epsilon = cv2.arcLength(curve = contour, closed = True) * arcPercentage
        corners = cv2.approxPolyDP(curve = contour, epsilon = epsilon, closed = True)
        x, y, w, h = cv2.boundingRect(points = corners)

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

    # Adaptive Thresholding requires the blocksize to be odd and bigger than 1
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
