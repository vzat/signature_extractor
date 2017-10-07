# import the necessary packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

# Opening an image from a file
f = easygui.fileopenbox()
I = cv2.imread(f)
Original = I.copy()

hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
h = hsv[:, :, 0]

img = cv2.adaptiveThreshold(h, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 5, C = 15)

cv2.imshow('th', img)

key = cv2.waitKey(0)
