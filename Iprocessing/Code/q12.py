# Question 12
import cv2
# Importing the Opencv library
img = cv2.imread('/home/whoami/Pictures/sharpor.png')
# Opening the Image file and assigning to a Matrix
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Original Image", img)  # Displaying the Original Image
img_lap = cv2.Laplacian(img, cv2.CV_64F)
# Using Laplacian Filter
cv2.namedWindow("Laplacian",cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Laplacian", img_lap)
# Displaying the image after Laplacian filter
final_image = img - 0.5*img_lap
# Subtracting the fraction of Laplacian Filter output from the input image
# to get the sharpened image
final_image = cv2.convertScaleAbs(final_image)
# This function will convert the image from 64 bit Float to 8 bit Unsigned image
cv2.namedWindow("Sharpened image",cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Sharpened image", final_image)
#  Displaying Image
cv2.waitKey()  # Wait for a key stroke from the User
