import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
# from models.CSRNet import CSRNet
from mcnn_model import MCNN
from torch.utils.data import DataLoader
from v_dataloader import CityCam
# from counting_datasets.CityCam_maker import  
from v_dataloader import ToTensor
# from my_dataloader import CrowdDataset
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
import os
import cv2
sys.path.append('..')

from PIL import Image
import scipy.spatial
import scipy.ndimage
from sklearn.preprocessing import MinMaxScaler

from scipy.ndimage import gaussian_filter

# Global variables
use_gpu = torch.cuda.is_available()
checkpoint_path = r'D:\Soumi\Soumi DI\DmapVehicleDetection\checkpoints\CSRNet-Epochs-10_BatchSize-64_LR-0.0001_Momentum-0.95_Gamma-0.5_Version-1\best_model.pt'
device = torch.device('cuda:0' if use_gpu else 'cpu')

def infer(sample):
    model = MCNN()
    # checkpoint = torch.load(checkpoint_path) #28.9.22
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) #28.9.22// run with only cpu
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    img = sample
    # print("Image shape:",img.shape)
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device)
    
    with torch.set_grad_enabled(False):
        et_dm = model(img).detach()
        et_dm = et_dm.squeeze(0).cpu().numpy()
   
    x=et_dm.squeeze()
    x = gaussian_filter(x, sigma=3)

    return x

data_trans = transforms.Compose([transforms.ToTensor()])
imgpath=r'D:\Soumi\Test Sample/Cars197_comp2.png'
readimg=cv2.imread(imgpath)
gray_img=Image.open(imgpath).convert('RGB')
h,w=gray_img.size
img= cv2. cvtColor(readimg, cv2.COLOR_BGR2RGB)
print(h,w)

img=data_trans(img)
print('Image shape',img.shape)
dm_image=infer(img)
plt.imshow(dm_image,cmap=CM.jet_r)
plt.show()
DMwidth, DMheight = dm_image.shape
print(dm_image.shape)


##############################      K-Means Clustering of Density map iteration 2     ############################################

pre_dm_path="D:\Soumi\Test Sample/Cars197_dm2.png"
pre_dm_IMG=Image.open(pre_dm_path).convert('L')
# pre_dm = pre_dm.resize((h,w))
pre_dm= np.asarray(pre_dm_IMG)
plt.imshow(pre_dm, cmap='gray')
plt.show()
print(pre_dm.shape)

dmwhite= np.argwhere(pre_dm>0)
# print(dmwhite)

pixels=[]

# h,w=OriginalImg.size
for i in dmwhite:   
    coordinate=x,y=i[0],i[1]
    # print(coordinate)
    # print(OriginalImg.getpixel(coordinate))
    # print(x,y)
    pixel_value=dm_image.item((x,y))
    image_pixel_value=(gray_img.getpixel((x,y)))
    
    # print(image_pixel_value)
    pixels.append(pixel_value)
   
    # pixels.append(OriginalImg.getpixel(coordinate))
    #print(pixel_value)

pixels=np.array(pixels)
print(pixels.shape)


x1d = pixels.copy().reshape(-1, 1)
# print("Values:",x1d)
print(x1d.shape)


kmeans = KMeans(n_clusters=2, random_state=0, max_iter= 10).fit(x1d)
y_pred=kmeans.predict(x1d)
labels=kmeans.labels_

# print(labels)
centroids = kmeans.cluster_centers_
a2 = centroids[labels]
# a2= labels*255
bin_img = np.where(a2== np.min(a2), 255, 0)

j=0
for u,v in dmwhite:
    pre_dm_IMG.putpixel((v,u),int(bin_img[j]))
    j=j+1

plt.imshow(pre_dm_IMG)
plt.show()
plt.imsave("D:\Soumi\Test Sample/Cars197_dm3.png",pre_dm_IMG, cmap='gray')


new_dm=Image.open("D:\Soumi\Test Sample/Cars197_dm3.png").convert('L')
new_dm=new_dm.resize((h,w))
blank = gray_img.point(lambda _: 0)
# comp=Image.composite(OriginalImg,blank,pil_image)
comp=Image.composite(gray_img,blank,new_dm)
plt.imshow(comp)
plt.show() 
comp.save(r'D:\Soumi\Test Sample/Cars197_comp3.png')


################################## Mean and STD count ###################################################

dm_array=np.asarray(new_dm)
maskwhite= np.argwhere(dm_array>0)
image_pixels=[]
# h,w=OriginalImg.size
for i in maskwhite:   
    coordinate=x,y=i[0],i[1]
    # print(coordinate)
    # print(OriginalImg.getpixel(coordinate))
    # print(x,y)
    
    image_pixel_value=(gray_img.getpixel((y,x)))
    
    # print(image_pixel_value)
    
    image_pixels.append(image_pixel_value)
    # pixels.append(OriginalImg.getpixel(coordinate))
    #print(pixel_value)

# print(image_pixels)
print(len(maskwhite))
# print(image_pixels)
# print("Count of Restored pixel values",len(image_pixels))
# print("Restored pixel values",image_pixels)
# histog, bin_edges = np.histogram(image_pixels, bins=256, range=(0, 255))
# plt.plot(histog) 
# plt.show()
print("mean value",np.mean(image_pixels))
print("std value", np.std(image_pixels))

