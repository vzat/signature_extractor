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
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(G, 50, 200)

cv2.imshow('canny', canny)

key = cv2.waitKey(0)
