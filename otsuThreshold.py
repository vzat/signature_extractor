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

img = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", img)

t, i = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("w/o Blur", i)

blur = cv2.GaussianBlur(img,(5,5),0)
t2, i2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("w Blur", i2)

key = cv2.waitKey(0)
