import cv2
import numpy as np
import os
import re
import tqdm
from tqdm import tqdm
folder='Flame'
images=[]
filename1 = "segmented_imgs/img/Fire/" ## something that changes in this loop -> you can set a complete path to manage folders
i=0
def count_bright_pixels(hsv, threshold):
    # convert to HSV color space
    count=0
    for i in hsv[:,:,2]:
        for j in i:
            if j>threshold:
                count+=1
    # create a binary mask for pixels with brightness greater than the threshold
    # mask = hsv[:,:,2] > threshold
    
    # count the number of pixels with brightness greater than the threshold
    #count = np.sum(mask)
    
    return count



# def big_jump(img):
#     max = 0
#     thres = 0
#     img = img.flatten()
#     img.sort()
#     list1 = []
    
#     for i in range(img.shape[0]-1):
#         if (img[i+1]!=img[i]):
#             # max = img[i+1]-img[i]
#             # thres = img[i]
#             list1.append(i)
#     return len(list1)

def custom_inRange(frame,lower,upper):
    #tries to bring white pixels in the mask
    output = np.zeros((540, 960)).astype(np.uint8)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i][j][0] >= lower[0] and frame[i][j][0] <= upper[0]:
                if frame[i][j][1] >= lower[1] and frame[i][j][0] <= upper[1]:
                    if frame[i][j][2] >= lower[2] and frame[i][j][0] <= upper[2]:
                        output[i][j] = 255
            if frame[i][j][2] >= 250 and frame[i][j][0] <= 60:
                output[i][j] = 255



    
    return output


low_green = np.array([89, 200, 200])
high_green = np.array([89, 255, 255])
# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
#     return sorted(data, key=alphanum_key)


# non_masked_images = os.listdir(input_folder)
# non_masked_images = sorted_alphanumeric(non_masked_images)
# for img in tqdm(non_masked_images):
#         cv_image = cv2.imread(folder+"/"+img)
#         if cv_image is not None:
#             # cv_image = image = cv2.imread('image1.jpg')
#             frame = cv2.resize(cv_image, (960, 540))
#             rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#             #blur = cv2.GaussianBlur(frame, (21, 21), 0)
#             hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


            

#             # big_jump_vars = big_jump(hsv[:,:,2])

#             avg_brightness = np.mean(hsv[:,:,2])
#             top25_brightness = np.percentile(hsv[:,:,2],75)
#             top20_brightness = np.percentile(hsv[:,:,2],80)
#             top10_brightness = np.percentile(hsv[:,:,2],90)
#             bright_pixel_count = count_bright_pixels(hsv, 254)



#             ratio = (avg_brightness / bright_pixel_count)
#             upper_v=255
            

#             lower = [0, 70, top25_brightness]
#             upper = [35, 255, 255]



#             lower = np.array(lower, dtype="uint8")
#             upper = np.array(upper, dtype="uint8")

#             mask = custom_inRange(hsv, lower, upper)
#             #print(i,ratio,top25_brightness,frame.shape, mask.shape)



#             output = cv2.bitwise_and(frame, hsv, mask=mask)
#             # mask2 = cv2.inRange(hsv, low_green, high_green)
#             # # inverse mask
#             # # mask2 = 255-mask2
#             # res = cv2.bitwise_and(output, output, mask=mask2)
#             # height, width, _ = output.shape



#             # ret,thresh = cv2.threshold(mask, 40, 255, 0)
#             # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#             # if len(contours) != 0:
#             #     # draw in blue the contours that were founded
#             #     c = max(contours, key = cv2.contourArea)
#             #     cv2.drawContours(output, c, -1, 0, 3)

#             # #cv2.fillPoly(output, pts =[c], color=(255,255,255))
#             # for i in range(height):
#             #     for j in range(width):
#             #         # img[i, j] is the RGB pixel at position (i, j)
#             #         # check if it's [0, 0, 0] and replace with [255, 255, 255] if so
#             #         if output[i, j].sum() >0 and output[i, j].sum() <765:
#             #             output[i, j] = [0,0,0]  
                
#             cv2.imwrite(filename1+"Flame" + str(i)+ ".jpg", output)
#             i=i+1


cv_image = cv2.imread('/home/omkar/Downloads/Flame_cropped/')
if cv_image is not None:
    # cv_image = image = cv2.imread('image1.jpg')
    frame = cv2.resize(cv_image, (960, 540))
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


            

            # big_jump_vars = big_jump(hsv[:,:,2])

    avg_brightness = np.mean(hsv[:,:,2])
    top25_brightness = np.percentile(hsv[:,:,2],75)
    top20_brightness = np.percentile(hsv[:,:,2],80)
    top10_brightness = np.percentile(hsv[:,:,2],90)
    bright_pixel_count = count_bright_pixels(hsv, 254)



            #ratio = (avg_brightness / bright_pixel_count)
    upper_v=255
            
    lower = [0, 70, top25_brightness]
    upper = [35, 255, 255]



    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = custom_inRange(hsv, lower, upper)
            #print(i,ratio,top25_brightness,frame.shape, mask.shape)



    output = cv2.bitwise_and(frame, hsv, mask=mask)
            # mask2 = cv2.inRange(hsv, low_green, high_green)
            # # inverse mask
            # # mask2 = 255-mask2
            # res = cv2.bitwise_and(output, output, mask=mask2)
            # height, width, _ = output.shape



            # ret,thresh = cv2.threshold(mask, 40, 255, 0)
            # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # if len(contours) != 0:
            #     # draw in blue the contours that were founded
            #     c = max(contours, key = cv2.contourArea)
            #     cv2.drawContours(output, c, -1, 0, 3)

            # #cv2.fillPoly(output, pts =[c], color=(255,255,255))
            # for i in range(height):
            #     for j in range(width):
            #         # img[i, j] is the RGB pixel at position (i, j)
            #         # check if it's [0, 0, 0] and replace with [255, 255, 255] if so
            #         if output[i, j].sum() >0 and output[i, j].sum() <765:
            #             output[i, j] = [0,0,0]  
                
    cv2.imwrite(filename1+"1Flametest" + str(i)+ ".jpg", output)
            #i=i+1
   
# cv2.imshow("window",output)
# cv2.waitKey()
# cv2.imwrite('filename2.jpg', output)