# Question 4
import cv2
# Importing Opencv libraries
import numpy as np
# Importing numpy module which will support all arithemetic operations on arrays
img = cv2.imread('/home/whoami/Pictures/assign1.jpg')
# Opening an Image and assigning it to a matrix
cv2.imshow('Original Image', img)  # Displaying Original image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Changing the image from RGB to GrayScale
cv2.imshow('GrayScale Image', img_gray)  # Displaying the Grayscale image
cv2.waitKey()  # Wait for a keystroke from the user
