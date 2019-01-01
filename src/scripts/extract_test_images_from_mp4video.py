#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv
import json
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import shutil
import time as time


# In[2]:

train=False
## paths
speed_txt_path = '../data/test/files/driving.txt'
speed_csv_path = '../data/test/files/driving.csv'
gen_img_path = '../data/test/imgs/'
mp4_path = '../data/test/videos/drive.mp4'
input_csv_path = '../data/test/files/driving_test.csv'


# In[4]:

if train:
    with open(speed_txt_path) as speed_file:
        speed_ground=speed_file.readlines()
    print(len(speed_ground))
    # In[5]:

speed_df = pd.read_csv(input_csv_path)
print("first few lines.....")
print(speed_df.head(5))
#len(speed_df)


# In[6]:


# load sub-set of images for code debug purposes  ****** DEBUG ONLY ******** COMM
#speed_df=speed_df.head(322) #161


# In[7]:


# VIDEO TO (IMAGES AND CSV FILE)
# only if images have not been extracted then this code block gets executed to generate jpgs from mp4
# make sure that 'gen_img_path' has no jpgs if running for first time 

def write_image_to_disk(idx, cap, writer, item):
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    
    #read in the image
    success, image = cap.read()
    print success

    if success:
        image_path = os.path.join(gen_img_path, str(idx) + '.jpg')

        #save image to IMG folder
        cv2.imwrite(image_path, image)

        #write row to train1.csv
        writer.writerow({'image_path': image_path,
                  'frame': idx,
                  'speed':float(item),
                 })
        





# In[8]:


last_count = 0
if not any(fname.endswith('.jpg') for fname in os.listdir(gen_img_path)):

    with open(speed_csv_path, 'w') as csvfile:
         fieldnames = ['image_path', 'frame', 'speed']
         writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
         writer.writeheader()

         #Path to raw image folder
         abs_path_to_IMG = os.path.join(gen_img_path)

         cap = cv2.VideoCapture(mp4_path)
         cap.set(cv2.CAP_PROP_FRAME_COUNT, len(speed_df))

         for idx, item in speed_df.iterrows():
            print('idx >> ', idx)
            print('item >> ', item['speed'])
            
            if idx % 100 == 0:
                print idx , 'images extracted .... '
            
            write_image_to_disk(idx, cap, writer, item['speed'])
        
         last_count = idx
         
                
            
print('images extracted..')


# In[9]:


## take care of regenerating the last image 1 more time to fix the pairs of image count

if last_count != 0 and idx%2 != 0 :
    idx +=1
    print idx
    shutil.copyfile(gen_img_path+ str(idx-1) + '.jpg', gen_img_path + str(idx) + '.jpg')


# In[10]:
