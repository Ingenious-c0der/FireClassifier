import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('Fire/fire.653.png')



# Convert the image to HSV format
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for yellow color in HSV
lower_yellow = np.array([20, 100, 200])
upper_yellow = np.array([30, 255, 255])
lower_orange = np.array([5, 50, 200])
upper_orange = np.array([10, 255, 255])

# Threshold the image to extract orange regions
mask1 = cv2.inRange(hsv, lower_orange, upper_orange)
# Threshold the image to extract yellow regions
mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Count the number of yellow pixels for each x-coordinate
pixel_count1 = np.sum(mask1, axis=0)
pixel_count2= np.sum(mask2, axis=0)
# final pixel count
pixel_count = pixel_count1+pixel_count2

# Plot the results
plt.plot(pixel_count)
plt.xlabel('X-coordinate')
plt.ylabel('Number of yellow/org pixels')
plt.show()
