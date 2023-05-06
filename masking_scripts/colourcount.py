import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('Fire/fire.654.png')
img= cv2.imread('Flame_cropped/burning-candles-on-dark-wooden-260nw-134578730.jpg')
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
        if np.mean(pixel_count[i:i+2]) > 500 and np.mean(pixel_count[i-2:i]) < 10:
            spike_indices.append(i)
        elif np.mean(pixel_count[i:i+2]) < 2 and np.mean(pixel_count[i-2:i]) > 500:
            fall_indices.append(i)

if (std_dev<1200):
    if len(spike_indices)==len(fall_indices)==0:
        
        print("fire detected")
    elif abs(len(spike_indices)-len(fall_indices))>2*min(len(spike_indices),len(fall_indices)):
        
        print("fire detected") 
    else:
        
        print("flame detected")
else:
        
        print("fire detected")

plt.plot(pixel_count)
plt.xlabel('X-coordinate')
plt.ylabel('Number of yellow pixels')
plt.title('Spike Indices: {}, Fall Indices: {}'.format(spike_indices, fall_indices))
plt.show()
print("std dev",std_dev)
print("spike count",len(spike_indices))
print("fall count",len(fall_indices))