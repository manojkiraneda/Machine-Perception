import cv2
# Importing the opencv library

img = cv2.imread("C:\\Users\Ram\Pictures\\nothuman\\24063.jpg", cv2.IMREAD_GRAYSCALE)
# Read the image as grayscale image
cv2.namedWindow("Original Image in GrayScale", cv2.WINDOW_NORMAL)
# Creating a window to display image
cv2.imshow("Original Image in GrayScale", img)  # Displaying the image
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Applying the sobel operator in the x direction
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Applying the sobel operator in the Y direction
sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, dtype=cv2.CV_8U)
#  Weighted average of sobel_x and sobel_y
sobel = cv2.bitwise_not(sobel)  # Inverting the sobel output
canny = cv2.Canny(img, 110, 160)  # Applying the canny operation
canny_inverter = cv2.bitwise_not(canny)  # Inverting the canny output
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Constructing the ellipse kernel for morphological operation
morph = cv2.dilate(canny, kernel, iterations=1)
# Applying the morphological operation-dilate to strengthen the edges
morph = cv2.bitwise_not(morph)  # Inverting the output
cv2.namedWindow("Canny + Morphological operation", cv2.WINDOW_NORMAL)
# Creating a window to display image
cv2.imshow("Canny + Morphological operation", morph)  # Display image
cv2.namedWindow("Canny Edge Detection Result", cv2.WINDOW_NORMAL)
# Creating a window to display image
cv2.imshow("Canny Edge Detection Result", canny_inverter)  # Display image
cv2.namedWindow("Sobel Edge Detection result", cv2.WINDOW_NORMAL)
# Creating a window to display image
cv2.imshow("Sobel Edge Detection result", sobel)  # Display image
cv2.waitKey()  # wait for the user keystroke
