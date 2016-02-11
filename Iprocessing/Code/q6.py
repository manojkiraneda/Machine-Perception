# Question 6
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
# Changing image from RGB to YCRCB model because it is easy to process
# image in that model
cv2.namedWindow("YCR_CB Image", cv2.WINDOW_NORMAL)
# Naming the window that contains image and making the user to resize the window
cv2.imshow("YCR_CB Image", img_ycrcb)  # Displaying the image
Threshold_Min = np.array([0, 130, 77])  # Setting minimum threshold for Y,Cr,Cb
Threshold_Max = np.array([255, 175, 127])  # Setting maximum threshold for Y,Cr,Cb
skinMask = cv2.inRange(img_ycrcb, Threshold_Min, Threshold_Max,img_ycrcb)
#  Getting a binary image based on the two threshold images
cv2.namedWindow("Skin_B and W", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Skin_B and W", skinMask)  # Displaying the skin mask
notskin = cv2.bitwise_not(skinMask)  # Calculating image which is not the skin
cv2.namedWindow("Not Skin_B and W", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Not Skin_B and W", notskin)  # Displaying image which is not the skin
skinMaskcolor = cv2.bitwise_or(img, img, mask=skinMask)
# Masking the skin mask on the original image to get the colored skin
cv2.namedWindow("Skin in Color", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Skin in Color", skinMaskcolor)  # Displaying the image
img_h, img_s, img_v = cv2.split(cv2.cvtColor(skinMaskcolor, cv2.COLOR_RGB2HSV))
#  Converting the image from RGB to HSV to improve the skin
img_v = cv2.multiply(img_v, 0.7)
#  Increasing the value of the intensity of the skin
skinMaskcolor = cv2.merge([img_h, img_s, img_v])
#  Merging the H,S,V channels
skinMaskcolor = cv2.cvtColor(skinMaskcolor, cv2.COLOR_HSV2RGB)
#  Converting the HSV image back to RGB
cv2.namedWindow("Skin in Color Enhanced", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Skin in Color Enhanced", skinMaskcolor)  # Displaying the enhanced image
notskinMaskcolor = cv2.bitwise_or(img, img, mask=notskin)
#  OR ing the image with itself with notskin as a Mask
cv2.namedWindow("Not Skin in Color", cv2.WINDOW_NORMAL)  # Naming the window
cv2.imshow("Not Skin in Color", notskinMaskcolor)  # Displaying the image other than the skin
final_image = cv2.bitwise_or(skinMaskcolor, notskinMaskcolor, mask=None)
#  adding both the skin and the image having other than skin
cv2.namedWindow("Final_Fairer", cv2.WINDOW_NORMAL)  # Naming the image
cv2.imshow("Final_Fairer", final_image)  # Displaying the final enhanced image
cv2.waitKey()  # Wait for a keystroke from the user
