import numpy as np
import cv2
import easygui

def readImage():
    file = easygui.fileopenbox()
    img = cv2.imread(file)

    if img is None:
        print 'Error: Not a valid image type'

    return img

img = readImage()

# buttonbox?
# Buttons:
# select file
# capture image from camera
# export signature to file
# display signature
# maybe extra features (checkboxes?) to improved the detection

cv2.imshow('Test', img)
cv2.waitKey(0)
