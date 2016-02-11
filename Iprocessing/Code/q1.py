# Question 1
import cv2
# Importing the Opencv library
img = cv2.imread("/home/whoami/Pictures/assign1.jpg")
# Reading the image and storing it in a matrix
cv2.imshow("Original Image", img)  # Displaying the Original Image
img_r = img[:, :, 0]  # Assigning the first channel to another Matrix
img_g = img[:, :, 1]  # Assigning the Second channel to another Matrix
img_b = img[:, :, 2]  # Assigning the Third channel to another Matrix
cv2.imshow("R_Component", img_r)  # Displaying the image
cv2.imshow("G_Component", img_g)  # Displaying the image
cv2.imshow("B_Component", img_b)  # Displaying the image
cv2.waitKey()  # Wait for a Key stroke from the User
