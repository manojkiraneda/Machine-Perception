# Question 3
import cv2
# Importing Opencv libraries
img = cv2.imread('/home/whoami/Pictures/assign1.jpg')
# Opening an Image and assigning it to a matrix
cv2.imshow('Original Image', img)  # Displaying Original image
img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  # Changing the image from RGB to LAB Model
cv2.imwrite("/home/whoami/new.jpg",img_lab)
cv2.imshow('L*a*b* Model', img_lab)  # Displaying the image
cv2.waitKey()  # Wait for a keystroke from the user
