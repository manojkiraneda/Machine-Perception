# Question 9
import cv2
# Importing the Opencv library
import numpy as np
# Importing NumPy,which is the fundamental package for scientific computing with Python
import random
# Importing random library
# Defining a noise generating function
def MakeSaltAndPepperNoise (Image, SaltNum, PepperNum):
 CopyImage = Image.copy()  # Taking one more copy of image
 nChannel = 0  # initialise the channel number to 0
 width = CopyImage.shape[0]  # Getting the width of image
 Height = CopyImage.shape[1]  # Getting the height of image
# If image is grayscale, it not have Image.shape[2]
# so it raise IndexError exception
 try:
  nChannel = CopyImage.shape[2]
 except IndexError:
  nChannel = 1
# Make Salt Noise
 for Salt in range(0, SaltNum):
# Generate Random Position
  RWidth = random.randrange(0, width)
  RHeight = random.randrange(0, Height)
# Make Noise
  if nChannel > 1:
   for c in range(0, nChannel):
    CopyImage[RWidth, RHeight, c] = 255
  else:
      CopyImage[RWidth, RHeight] = 255
# Make Pepper Noise
 for Pepper in range(0, PepperNum):
  # Generate Random Position
  RWidth = random.randrange(0, width)
  RHeight = random.randrange(0, Height)
  # Make Noise
  if nChannel > 1:
   for c in range(0, nChannel):
    CopyImage[RWidth, RHeight, c] = 0
  else:
   CopyImage[RWidth, RHeight] = 0
 return CopyImage  # Return image which has salt and pepper noise
img = cv2.imread("/home/whoami/Pictures/kid.jpg")
# Reading an image
cv2.imshow("Original Image", img)  # Displaying an image
new_img = MakeSaltAndPepperNoise(img, 1000, 1000)
# Adding Salt and Pepper Noise to the image
cv2.imshow("Noise added Image", new_img)
# Displaying the noisy image
new_img1 = cv2.medianBlur(new_img, 5, 0)
# Applying median blur to the noisy image
cv2.imshow("Noise Cleared", new_img1)
# Display noise removed image
cv2.waitKey()  # Wait for a keystroke from the user
