#!/usr/bin/env python
# coding: utf-8
# ### 2. Video Features
# 
# + `extract_vgg16_relu6.py`  
#   + used to extract video features  
#      + Given an image (size: 256x340), we get 5 crops (size: 224x224) at the image center and four corners. The `vgg16-relu6` features are extracted for all 5 crops and subsequently averaged to form a single feature vector (size: 4096).  
#      + Given a video, we process its 25 images seuqentially. In the end, each video is represented as a feature sequence (size: 4096 x 25).  
#   + written in PyTorch; supports both CPU and GPU.  
# 
# + `vgg16_relu6/`  
#    + contains all the video features, EXCEPT those belonging to class 1 (`ApplyEyeMakeup`)  
#    + you need to run script `extract_vgg16_relu6.py` to complete the feature extracting process   
# 
# In[4]:


# write your codes here
import torch
import torch.nn as nn
import torch.utils.data as data_utils

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import glob
import scipy.io
from PIL import Image
import torchvision.models as models
from torch.optim import lr_scheduler
import os
import numpy as np
import h5py

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as DD
import torchnet as tnt

import time


# In[2]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
prep = transforms.Compose([ transforms.ToTensor(), normalize ])
#prep(img)


# In[3]:


class1_path = '../data/test/imgs/*'#'./UCF101_release/images_class1/*'
class1_path_new = '../data/test/imgs/' #'./UCF101_release/images_class1/'

vgg_gen_mat_path = '../data/test/mat/'
frame_names = [name[-8:] for name in glob.glob(class1_path )]
image_class1_paths = [name for name in glob.glob(class1_path)]
print(len(frame_names))
#print(frame_names)
#print(image_class1_paths)


# In[5]:


model_vgg16 = models.vgg16(pretrained=True) 
print("------------")
for param in model_vgg16.parameters():
    param.requires_grad = False

## outputs contain the 1st ReLU output of the classifier layer in the VGG 16 network as given in question
## we will outputs as vgg features to process the next steps in this problem
outputs= list()
def hook(module, input, output):
    outputs.append(output)

print(model_vgg16.classifier[1])
model_vgg16.classifier[1].register_forward_hook(hook)  #relu

print("------------")
print(model_vgg16.features)
print(model_vgg16.classifier)


# In[6]:


def get_vgg_feature(img):
    #features = vgg16_FC1(img)
    features = model_vgg16(img)
    return features
    

def load_crop_extract_save(image_filename):
    newImgTnsrs = dict()
    newImgTnsrs['Feature'] = list()
    #for image_filename in glob.glob(aImgClass1Path + '/*'):
        
    print("Processing image .... ",image_filename)
    # for each image crop 5 areas
    img = Image.open(image_filename)
    
    height = img.size[1]
    width = img.size[0]
    
    # tuple defining the left, upper, right, and lower pixel coordinates
    # training image is 640(w) X 480(h) ; test image is 640(w) X 840(h) =>
    # Train:: ignore 32 px from top - dont need skies as features.
    top_left = img.crop((0, 32, 224, 256))
    top_right = img.crop((width - 224, 32, width, 256))
    bottom_left = img.crop((0, height - 224, 224 , height))
    bottom_right = img.crop(( width - 224, height - 224, width, height))
    center =  img.crop(( (width/2) - 112, (height/2) - 112, (width/2) + 112, (height/2) + 112 ))
    
    #normalize and transform cropped images before passing to vgg 
    stcked_tnsr = torch.stack((prep(top_left), prep(top_right),prep(bottom_left) ,prep(bottom_right),prep(center)) , 0 )
    #print('stcked_tnsr ',stcked_tnsr.shape)
    stcked_vgg = get_vgg_feature(stcked_tnsr)
    print(stcked_vgg.shape)
    #print('output ',len(outputs))
    
    ## find mean
    new_img_mean = torch.mean(outputs[0], 0)
    #print('new_img_mean ',new_img_mean.shape)
    newImgTnsrs['Feature'].append(new_img_mean.numpy())
    #outputs.clear() ## so that next image 
    del outputs[:] #py 2.7 syntx
    
    #break ## comment out
    
    print('aImgTnsr >> ',len(newImgTnsrs['Feature']))
    last_dot_index = image_filename.rfind('.')
    last_slash_index = image_filename.rfind('/')
    mat_file_name = image_filename[last_slash_index + 1: last_dot_index] 
    scipy.io.savemat(vgg_gen_mat_path +mat_file_name, newImgTnsrs)
        
        
        
        
    


# In[2]:


################################### UNCOMMENT THE CODE BELOW TO RUN EXTRACT VGG FEATURES (MAT FILES) ################## 
## get image features for class 1 - extract features - UNCOMMENT if you want to extract features again for class1
##

##('extracting .. ', '../data/train/imgs/2586.jpg')
## ('Processing image .... ', '../data/train/imgs/2586.jpg')
##
start_time = time.time()


for aClass1Img in image_class1_paths: #glob.glob(image_class1_paths + '/*'):
    print("extracting .. ",aClass1Img)

    ## check if mat file already exists, else generate it.
    last_dot_index = aClass1Img.rfind('.')
    last_slash_index = aClass1Img.rfind('/')
    mat_file_name = aClass1Img[last_slash_index + 1: last_dot_index] 

    total_mat_path = vgg_gen_mat_path + mat_file_name+'.mat'
    print(total_mat_path)

    if total_mat_path not in glob.glob(vgg_gen_mat_path + '/*'):
        load_crop_extract_save(aClass1Img)
    else:
        print('mat file for %s already exists !!'%aClass1Img)
    
    #break  # comment out for all 
end_time = time.time() - start_time
print('------------------------------------------------------------')
print("time taken to extract features for Class 1  in seconds {:.3f}".format(end_time))

