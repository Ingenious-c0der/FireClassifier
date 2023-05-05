import os
from PIL import Image

# set the path to the folder containing the images to be cropped
input_folder = 'Flame'

# create a new folder to store the cropped images
output_folder = 'Flame_cropped'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# set the number of pixels to crop from the bottom of the images
bottom = 50

# loop through all the files in the input folder
for filename in os.listdir(input_folder):
    # open the image file using PIL
    with Image.open(os.path.join(input_folder, filename)) as img:
        # get the width and height of the image
        width, height = img.size
        
        # crop the image to remove the label at the bottom
        cropped_img = img.crop((0, 0, width, height - bottom))
        
        # save the cropped image to the output folder
        cropped_img.save(os.path.join(output_folder, filename))