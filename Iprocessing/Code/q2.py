# Question 2
import cv2
# Importing Opencv Libraries
img = cv2.imread('/home/whoami/Pictures/assign1.jpg')
# Opening the Image file and assigning to a Matrix
cv2.imshow('Original Image', img)  # Displaying Original Image
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Conversion of Image from RGB to HSV
img_hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)  # Conversion of image from RGB to HSL
cv2.imshow('HSV Model', img_hsv)  # Displaying Image in HSV Model
cv2.imshow('HSL Model', img_hsl)  # Displaying Image in HSL Model
img_h, img_s, img_v = cv2.split(img_hsv)  # Splitting the image into h,s,v Channels
img_h1, img_s1, img_l = cv2.split(img_hsl)  # Splitting the image into h,s,l Channels
cv2.imshow("H Component in HSV", img_h)  # Display Image
cv2.imshow("S Component in HSV", img_s)  # Display Image
cv2.imshow("V Component in HSV", img_v)  # Display Image
cv2.imshow("H Component in HSL", img_h1)  # Display Image
cv2.imshow("S Component in HSL", img_s1)  # Display Image
cv2.imshow("L Component in HSL", img_l)  # Display Image
cv2.waitKey()  # Wait for a Keystroke
