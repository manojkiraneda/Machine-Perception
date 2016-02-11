# Question 10
import cv2
# Importing the Opencv library
import numpy as np
# Importing NumPy,which is the fundamental package for scientific computing with Python
img = cv2.imread("/home/whoami/Pictures/build.jpg", cv2.IMREAD_GRAYSCALE)
# Reading the image in grayscale and storing it in a matrix
sobelx_img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# Applying the Sobel Filter to the image in X direction
sobely_img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# This function is used to convert the image from float to 8 bit unsigned char
sobelx_img = cv2.convertScaleAbs(sobelx_img)
# This function is used to convert the image from float to 8 bit unsigned char
sobely_img = cv2.convertScaleAbs(sobely_img)
# Applying the Sobel Filter to the image in Y direction
prewit_ker_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
# Preparing the Prewit Filter Kernel in X direction
prewit_ker_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
# Preparing the Prewit Filter Kernel in Y direction
prewit_img_x = cv2.filter2D(img, cv2.CV_16S, prewit_ker_x)
# Applying the Prewit Filter to the image in X direction
prewit_img_y = cv2.filter2D(img, cv2.CV_16S, prewit_ker_y)
# Applying the Prewit Filter to the image in Y direction
prewit_img_x1 = cv2.convertScaleAbs(prewit_img_x)
# This function is used to convert the image from float to 8 bit unsigned char
prewit_img_y1 = cv2.convertScaleAbs(prewit_img_y)
# This function is used to convert the image from float to 8 bit unsigned char
final_prewit = cv2.addWeighted(prewit_img_x1, 0.5, prewit_img_y1, 0.5, 0)
# Taking the average of two images i.e X direction and Y direction
final_sobel = cv2.addWeighted(sobelx_img, 0.5, sobely_img, 0.5, 0, dtype=cv2.CV_8U)
# Taking the average of two images i.e X direction and Y direction
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", img)  # Displaying the Image
cv2.namedWindow("Prewit_x", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Prewit_x", prewit_img_x1)  # Displaying the image
cv2.namedWindow("Prewit_y", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Prewit_y", prewit_img_y1)  # Displaying the image
cv2.namedWindow("Prewit Filter", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Prewit Filter", final_prewit)  # Displaying the image
cv2.namedWindow("Sobel_x", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow('Sobel_x', sobelx_img)  # Displaying the image
cv2.namedWindow("Sobel_y", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow('Sobel_y', sobely_img)  # Displaying the image
cv2.namedWindow("Sobel Filter", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Sobel Filter", final_sobel)  # Displaying the image
cv2.waitKey()  # Wait for a Keystroke from the user
