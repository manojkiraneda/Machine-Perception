# Question 13
import cv2
# Importing the Opencv library
img = cv2.imread("/home/whoami/Pictures/assign1_1.jpg")
# Reading the image and storing it in a matrix
cv2.imshow("Original Image", img)  # Displaying the original Image
gau_img1 = cv2.GaussianBlur(img, (11, 11), 0)  # Using the Gaussian Filter First
lap_img2 = cv2.Laplacian(gau_img1, cv2.CV_64F, ksize=11)
#  Using the Laplacian filter after Gaussian
lap_img1_1 = cv2.Laplacian(img, cv2.CV_64F, ksize=11)
#  Using the Laplacian Filter first
gau_img1_2 = cv2.GaussianBlur(lap_img1_1, (11, 11), 0)
#  Using the Gaussian Filter Second
cv2.imshow("Gaussian Filter - First", gau_img1)  # Displaying image
cv2.imshow("Laplacian Filter - Second", lap_img2)  # Displaying image
cv2.imshow("Laplacian Filter - First", lap_img1_1)  # Displaying image
cv2.imshow("Gaussian Filter - Second", gau_img1_2)  # Displaying image
cv2.waitKey()  # Wait for a keystroke from user
