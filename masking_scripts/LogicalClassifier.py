import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,precision_score,recall_score
import re
input_folder='combined_fire_flame'
# output_folder='Flame_cropped_masked'
images=[]
flame_dir = "Flame_cropped/"
fire_dir = "Fire/"

def prep_img(file_path):
    IMG_SIZE = 150  # 50 in txt-based
    img_array = cv2.imread(file_path)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
#filename1 = "segmented_imgs/img/img75/" # something that changes in this loop -> you can set a complete path to manage input_folders


lower_yellow = np.array([20, 100, 200])
upper_yellow = np.array([30, 255, 255])
lower_orange = np.array([5, 50, 200])
upper_orange = np.array([10, 255, 255])
# def count_bright_pixels(hsv, threshold):
#     # convert to HSV color space
#     count=0
#     for i in hsv[:,:,2]:
#         for j in i:
#             if j>threshold:
#                 count+=1

y_pred = []
y_true = []

def getArclength(img):
    number_of_nonblackpixels = np.count_nonzero(img[:,:,2])
    contours, hierarchy = cv2.findContours(img[:,:,2],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    total_arclength = 0
    for i in contours:
        total_arclength += cv2.arcLength(i,True)
    return total_arclength

    
def predict(cv_image):
    
    rgb = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
    #blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Define the lower and upper bounds for yellow color in HSV


    # Threshold the image to extract orange regions
    mask1 = cv2.inRange(hsv, lower_orange, upper_orange)
    # Threshold the image to extract yellow regions
    mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Count the number of yellow pixels for each x-coordinate
    pixel_count1 = np.sum(mask1, axis=0)
    pixel_count2= np.sum(mask2, axis=0)
    # final pixel count
    pixel_count = pixel_count1+pixel_count2
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
    
    if abs(len(spike_indices)-len(fall_indices))>1:
        if(std_dev<1000):
            y_pred.append(0)
            print("flame detected")
        else:
            y_pred.append(1)
            print("fire detected")
    else:
        y_pred.append(0)
        print("flame detected")


flame_files = os.listdir(flame_dir)
fire_files = os.listdir(fire_dir)

for filename in flame_files:

    img = prep_img(flame_dir + filename)
    img_array = cv2.imread(flame_dir + filename)
    predict(img_array)
    y_true.append(0)  # 0 represents "flame" class

for filename in fire_files:
    img = prep_img(fire_dir + filename)
    img_array = cv2.imread(fire_dir + filename)
    predict(img_array)
    y_true.append(1)  # 1 represents "fire" class

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Predict the classes of the images


# Compute and plot the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
print("Precision: ",precision_score(y_true, y_pred))
print("Recall: ",recall_score(y_true, y_pred))
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

