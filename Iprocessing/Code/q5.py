# Question 5
import cv2
# Importing the Opencv library
import numpy as np
# Importing NumPy,which is the fundamental package for scientific computing with Python
img = cv2.imread('/home/whoami/Pictures/s6.jpg')
# Reading the image and storing it in a matrix
cv2.namedWindow("original image", cv2.WINDOW_NORMAL)
# Naming the window that contains image and making the user to resize the window
cv2.imshow("original image", img)  # Displaying Image
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# Changing image from RGB to YCRCB model
cv2.namedWindow("YCR_CB Image", cv2.WINDOW_NORMAL)
# Naming the window that contains image and making the user to resize the window
cv2.imshow("YCR_CB Image", img_ycrcb)  # Displaying the image
Threshold_Min = np.array([0, 130, 77])  # Setting minimum threshold for Y,Cr,Cb
Threshold_Max = np.array([255, 175, 127])  # Setting maximum threshold for Y,Cr,Cb
#  Threshold_Min = np.array([0, 30, 60])  # Setting minimum threshold for Y,Cr,Cb
#  Threshold_Max = np.array([20, 150, 255])  # Setting maximum threshold for Y,Cr,Cb
skinMask = cv2.inRange(img_ycrcb, Threshold_Min, Threshold_Max,img_ycrcb)
#  Intensity value is set to 255 if it's value lies between two thresholds and
#  otherwise ,the pixel value is set to zero
skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
# Using Gaussian Blur to smooth the skinmask
skin = cv2.bitwise_and(img, img, mask=skinMask)
# show the skin in the image along with the mask
cv2.namedWindow("Skin Detected", cv2.WINDOW_NORMAL)
# Naming the window that contains image and making the user to resize the window
cv2.imshow("Skin Detected", skin)  # Displaying Image
cv2.waitKey()  # Wait for a Keystroke from user
