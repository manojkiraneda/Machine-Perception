# Question 8
import cv2
# Importing the Opencv library
img = cv2.imread("/home/whoami/Pictures/5.jpg")
# Opening the Image file and assigning to a Matrix
cv2.imshow("Original Image", img)  # Display Original Image
new_image = cv2.GaussianBlur(img, (11, 11), 0)  # Applying Gaussian Blur with a kernel of size 11 X 11
new_image1 = cv2.GaussianBlur(img, (3, 3), 0)  # Applying Gaussian Blur with a kernel of size 3 X 3
new_image2 = cv2.GaussianBlur(img, (7, 7), 0)  # Applying Gaussian Blur with a kernel of size 7 X 7
new_image3 = cv2.GaussianBlur(img, (9, 9), 0)  # Applying Gaussian Blur with a kernel of size 9 X 9
cv2.imshow("Gaussian Blur (11 X 11)", new_image)  # Display Image
cv2.imshow("Gaussian Blur (3 X 3)", new_image1)  # Display Image
cv2.imshow("Gaussian Blur (7 X 7)", new_image2)  # Display Image
cv2.imshow("Gaussian Blur (9 X 9)", new_image3)  # Display Image
cv2.waitKey()  # Wait for a Keystroke from the user
