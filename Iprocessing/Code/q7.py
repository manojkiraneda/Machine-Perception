# Question 7
import cv2
# Import Opencv Library
import numpy as np
# Importing NumPy,which is the fundamental package for scientific computing with Python
img = cv2.imread("/home/whoami/Pictures/3.jpg", cv2.IMREAD_GRAYSCALE)
new = img.copy()
# Read image as grayscale image
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
# Naming the window that contains image and making the user to resize the window
cv2.imshow("Original image", img)
# Display Image
mean, stddev = cv2.meanStdDev(img)  # Calculate the mean and standard deviation
rows, columns = img.shape  # Getting rows and columns of original image
img2 = np.array(img, dtype=np.float)  # converting image from unsigned integer to float
for i in range(rows):
    for j in range(columns):
        if (img[i, j] - mean + 127) > 255:
            img2[i, j] = 255  # if the value more than 255 round to 255
        elif (img[i, j]-mean+127) < 0:
            img2[i, j] = 0  # If the value is less than 0 then round to 0
        else:
            img2[i, j] = img[i, j]-mean+127  # else do the whitening
img2 = np.array(img2,dtype=np.uint8)  # Converting the image from float to unsigned integer
new_image1 = cv2.equalizeHist(new)  # Histogram Equalisation
new_image1 = cv2.convertScaleAbs(new_image1)
cv2.namedWindow("Whitening Output", cv2.WINDOW_NORMAL)
#  Naming the window that contains image and making the user to resize the window
cv2.imshow("Whitening Output", img2)
# Display Whitened image
cv2.namedWindow("Histogram Equalisation", cv2.WINDOW_NORMAL)
# Naming the window that contains image and making the user to resize the window
cv2.imshow("Histogram Equalisation", new_image1)
# Display Histogram Equalised image
cv2.waitKey()  # Wait for the keystroke from the user
