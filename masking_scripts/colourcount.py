import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('Fire/fire.653.png')
#img= cv2.imread('Flame_cropped/church-candles-red-yellow-transparent-260nw-84934867.jpg')
# Convert the image to HSV format
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for yellow and orange colors in HSV
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
pixel_count2 = np.sum(mask2, axis=0)
# final pixel count
pixel_count = pixel_count1 + pixel_count2

# Calculate the standard deviation of pixel_count values
std_dev = np.std(pixel_count)

# Detect sudden spikes in pixel_count values
spike_indices = []
fall_indices = []
for i in range(len(pixel_count)):
    if pixel_count[i] > 100 and np.mean(pixel_count[i-2:i]) < 10:
        spike_indices.append(i)
    elif pixel_count[i] < 10 and np.mean(pixel_count[i-1:i]) > 100:
        fall_indices.append(i)

# plt.plot(pixel_count)
# plt.xlabel('X-coordinate')
# plt.ylabel('Number of yellow pixels')
# plt.title('Spike Indices: {}, Fall Indices: {}'.format(spike_indices, fall_indices))
# plt.show()
print("std dev",std_dev)
print("spike count",len(spike_indices))
print("fall count",len(fall_indices))