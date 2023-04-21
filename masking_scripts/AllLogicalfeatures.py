import cv2 as cv
import math
import numpy as np
import os

def contourprep(img):
    img = img[:,:,2]
    img = cv.blur(img,(10,10))
    contours, hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    c_selected = contours[0]
    for i in contours:
        if len(c_selected)< len(i): 
            c_selected = i

    c = list(c_selected)
    c_sort_by_y = sorted(c,key = lambda x : x[0][1])
    c_sort_by_x = sorted(c,key = lambda x : x[0][0])
    return contours,c_sort_by_x,c_sort_by_y
    
    

    
def getBaseWidthAndHeight(arr:list):
    min_y = arr[len(arr)-1][0][1]
    max_y = arr[0][0][1]
    range = (min_y - max_y)/4
    min_x = arr[len(arr)-1][0][0]
    max_x = arr[len(arr)-1][0][0]
    #print('hehe',max_x,min_x,range,max_y,min_y,min_y-range)

    
    i = len(arr)-1
    while i >= 0:
        if(arr[i][0][1] < int(min_y-range)):
            break
        if(arr[i][0][0] > max_x):
            max_x = arr[i][0][0]
        if(arr[i][0][0] < min_x):
            min_x = arr[i][0][0]
        i = i -1
    midpoint = math.floor((min_x + max_x)/2)

    return {"width":max_x - min_x,"height":min_y-max_y,"base_midpoint":midpoint}



def getSymmetryScore(arr:list,base_midpoint):
    

    min_x = arr[len(arr)-1][0][0]
    max_x = arr[0][0][0]
    mean = base_midpoint
    left_area = 0
    right_area = 0

    for i in range(0,len(arr)):
        if arr[i][0][0] < mean:
            left_area += mean-arr[i][0][0]
        else:
            right_area += arr[i][0][0]-mean
    return abs(left_area-right_area)/(left_area+right_area)

def getAttrributes(img):
    h_mat = img[:,:,0]
    s_mat = img[:,:,1]
    v_mat = img[:,:,2]
    number_of_nonblackpixels = np.count_nonzero(v_mat)
    average_h = h_mat[np.nonzero(h_mat)].mean()
    average_s = s_mat[np.nonzero(s_mat)].mean()
    average_v = v_mat[np.nonzero(v_mat)].mean()
    return number_of_nonblackpixels,average_h,average_s,average_v

def getArclengthAreaRatio(img):
    number_of_nonblackpixels = np.count_nonzero(img[:,:,2])
    contours, hierarchy = cv.findContours(img[:,:,2],cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    total_arclength = 0
    for i in contours:
        total_arclength += cv.arcLength(i,True)

    return total_arclength/number_of_nonblackpixels

def whitePixelsAreaRatio(img):
    
    number_of_nonblackpixels = np.count_nonzero(img[:,:,2])
    return len(np.where(img[:,:,2]>250))/number_of_nonblackpixels

img = cv.imread('/home/omkar/FireClassifier/segmented_imgs/img/img75/Fire34.jpg')

countours,c_sort_by_x,c_sort_by_y=contourprep(img)







fire = getBaseWidthAndHeight(c_sort_by_y)
print(fire,getSymmetryScore(c_sort_by_x,fire["base_midpoint"]))
print('\n')
print(whitePixelsAreaRatio(img),getArclengthAreaRatio(img),getAttrributes(img))
print('\n')