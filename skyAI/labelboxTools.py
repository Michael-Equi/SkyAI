#!/usr/bin/env python
# coding: utf-8

# ## Program for preparing labeled data from label box

# In[1]:


# Import necessary packages
import numpy as np
import json
import sys
#download tool for http requests
get_ipython().system('conda install --yes --prefix {sys.prefix} requests')
import requests
import imageio


# ### Get the data from the json file pointed to by the filepath below

# In[ ]:


# read the command line input for the path to the json file with labels
filepath = "data.json"

mask_img = []
img = []
with open(filepath, "r") as read_file:
    data = json.load(read_file)
    string = json.dumps(data, indent=2)
    # print(string) # for visualizing the json better
    i = 0
    for item in data:
        masks = item['Masks']
        for layer in masks:
            url = masks[layer]
            im = imageio.imread(url)
            try:
                mask_img[i].append(im[:, :, 0])
            except:
                mask_img.append([])
                mask_img[i].append(im[:, :, 0])
            
        image_url = item['Labeled Data']
        img.append(imageio.imread(image_url))
        i+=1


# In[ ]:


#Turn mask_images into a numpy array
images = np.array(img)
mask_images = np.array(mask_img)
print(mask_images.shape)


# In[ ]:


#Make sure that data was collected ok

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#view the results of the test
fig=plt.figure(figsize=(25, 25))
for i in range(len(images)):
                
    fig.add_subplot(5, 6, i+1)
    plt.imshow(images[i])
    fig.add_subplot(5, 6, i+2) #first mask of every image
    plt.imshow(mask_images[i, 0])
    
    
plt.show()


# In[ ]:


#Concatenate the masks into a single high channel array and turn array values from 0-255 to 0-1
mask_images = np.moveaxis(mask_images, 1, 3)
mask_images = mask_images/255.0
print(mask_images.shape)


# In[ ]:


#resize images
import skimage.transform as trans

target_size = (512,512)

images_resized = np.zeros((len(images),target_size[0],target_size[1],3))
mask_images_resized = np.zeros((len(images),target_size[0],target_size[1],(len(mask_images[0,0,0]))))

print(images.shape)
print(mask_images.shape)

for i in range(len(images)):
    images_resized[i] = trans.resize(images[i],target_size)
    mask_images_resized[i] = trans.resize(mask_images[i],target_size)
    
print(images_resized.shape)
print(mask_images_resized.shape)


# In[ ]:


#Make sure that data is transformed ok

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ind = np.unravel_index(np.argmax(mask_images, axis=None), mask_images.shape)
print(mask_images[ind]) #make sure that the image has been converted to range 0.0-1.0

#view the results of the test
fig=plt.figure(figsize=(25, 25))
for i in range(len(images_resized)):
                
    fig.add_subplot(5, 6, i+1)
    plt.imshow(images_resized[i])
    fig.add_subplot(5, 6, i+2) #first mask of every image
    plt.imshow(mask_images_resized[i], vmin=0, vmax=1)
    #plt.imshow(mask_images_resized[i,:,:,0], cmap='gray', vmin=0, vmax=1)

    
plt.show()


# In[ ]:


#save the images and their masks
import os
import random #generate distribution to assign 80% of images to train and 20% to test

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

for i in range(len(images_resized)):
    if(random.randint(1,101) < 80):
        imageio.imwrite("train/" + str(i) + '.jpg', images_resized[i])
        imageio.imwrite("train/" + str(i) + '_mask.jpg', mask_images_resized[i])
    else:
        imageio.imwrite("test/" + str(i) + '.jpg', images_resized[i])
        imageio.imwrite("test/" + str(i) + '_mask.jpg', mask_images_resized[i])

