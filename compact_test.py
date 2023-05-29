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
checkpoint_path = r'C:\Users\CVPR\Soumi DI\DmapVehicleDetection\checkpoints\CSRNet-Epochs-10_BatchSize-64_LR-0.0001_Momentum-0.95_Gamma-0.5_Version-1\best_model.pt'
device = torch.device('cuda:0' if use_gpu else 'cpu')
def infer(sample):
    model = MCNN()
    # checkpoint = torch.load(checkpoint_path) #28.9.22
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) #28.9.22// run with only cpu
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    img = sample
    print("Image shape:",img.shape)
    # density_map = dm
    # gt_count = sample['gt_count']
    
    # make img's dimension (1, C, H, W)
    img = torch.unsqueeze(img, dim=0)
    # make the dimension of the density map (1, H, W)
    # density_map = torch.unsqueeze(density_map, dim=0)
   
    
    img = img.to(device)
    # density_map = density_map.to(device)
    # print("image is:", img.data>0)
    
    
    with torch.set_grad_enabled(False):
        et_dm = model(img).detach()
        # down_sample = nn.Sequential(nn.MaxUnpool2d(2), nn.MaxUnpool2d(2))
        # et_dm = down_sample(et_dm)
        
        et_dm = et_dm.squeeze(0).cpu().numpy()
        print("Estimated density map shape is:",et_dm.shape)
        print("Estimated density map is:",et_dm)
        print("Max value of Estimated Density Map is:",et_dm.max())
        print("Min value of Estimated Density Map is:",et_dm.min())
        # down_gt_dm = down_gt_dm.cpu().numpy()
        # et_dm_reshape = et_dm.copy().reshape(-1, 1)
    print(f'The integral of the estimated density map: {et_dm.sum()}')
    
    print(et_dm.dtype)
   
    x=et_dm.squeeze()
    x = gaussian_filter(x, sigma=3)

    return x

data_trans = transforms.Compose([transforms.ToTensor()])
imgpath=r'C:/Users/CVPR/Soumi DI/NewTestSamples/New folder/7.jpg'
readimg=cv2.imread(imgpath)
h,w,c=readimg.shape
img= cv2. cvtColor(readimg, cv2.COLOR_BGR2RGB)

img=data_trans(img)
print('Image shape',img.shape)
dm_image=infer(img)
plt.imshow(dm_image,cmap=CM.jet_r)
plt.show()
DMwidth, DMheight = dm_image.shape
print(DMwidth,DMheight)


##############################      K-Means Clustering of Density map     ############################################
x1d = dm_image.copy().reshape(-1, 1)
print("Values:",x1d)
kmeans = KMeans(n_clusters=2, random_state=0, max_iter= 10).fit(x1d)
y_pred=kmeans.predict(x1d)
labels=kmeans.labels_
print('Labels value:', labels)
centroids = kmeans.cluster_centers_
# a2 = centroids[labels]
# print('a2 shape ', a2.shape)
a3 = labels.reshape(DMwidth, DMheight)
print('type of a3', type(a3))
a3= a3*255
np.save("C:/Users/CVPR/Soumi DI/NewTestSamples/New folder/7_DM0", a3)
a3img=Image.fromarray(a3)
# bin_img = np.where(a3== np.min(a3), 255, 0)
# bin_img=bin_img.resize(h,w)
# density_map=Image.fromarray(bin_img)

plt.imshow(a3img)
plt.show()
plt.imsave("C:/Users/CVPR/Soumi DI/NewTestSamples/New folder/7_DM0.png", a3img, cmap='gray')

pix = list(a3img.getdata())
print('list',pix)
# density_map = density_map.resize((w,h))
# print(density_map.size)
# print(type(density_map))
# dm_array=np.asarray(density_map)

###############################    k-means of original image    ########################################
grayimg=cv2.cvtColor(readimg, cv2.COLOR_RGB2GRAY)
vectorized = grayimg.reshape((-1,1))
vectorized = np.float32(vectorized)
K = 2
attempts=10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
km_image = res.reshape((grayimg.shape))
plt.imshow(km_image, cmap=CM.gray)
# plt.title('Segmented Image when K = 2'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imsave("C:/Users/CVPR/Soumi DI/NewTestSamples/New folder/7_KM0.jpg",km_image, cmap=CM.gray)
km_array=np.asarray(km_image)

##############################      Fusion of DM and KM k-means     ######################################
a=r'C:/Users/CVPR/Soumi DI/NewTestSamples/New folder/7_KM0.jpg'
img1=Image.open(a)
img1 = img1.convert('L') 
img1= np.asarray(img1)
w,h=img1.shape

# b=r'C:/Users/CVPR/Soumi DI/NewTestSamples/New folder/7_DM0.npy'
# img2=Image.open(b)
# img2=img2.convert('L')
# img2 = img2.resize((h,w))
# img2= np.asarray(img2)
# num_zero = (img2 == 0).sum() #black
# num_one = (img2 > 0).sum() #white
# print('DM value count',num_zero,num_one)
# image2=Image.fromarray(img2)
# # img2.show()
# # img2.save(r'C:/Users/CVPR/Soumi DI/NewTestSamples/New folder/7_DM6.jpg')
# # print("KM & DM shape:", img1.shape, img2.shape) 

# f_array=np.bitwise_or(img1,img2)
# num_zeros = (f_array == 0).sum() #black
# num_ones = (f_array > 0).sum() #white
# print('Fusion value count',num_zeros,num_ones)
# # print(newimg)
# f_image=Image.fromarray(f_array)
# plt.imshow(f_image)
# plt.show()

# ######################################      pixel restore in Fusion image        #############################
# OriginalImg=Image.open(imgpath).convert('L')
# blank = OriginalImg.point(lambda _: 0)
# # comp=Image.composite(OriginalImg,blank,pil_image)
# comp_img=Image.composite(OriginalImg,blank,f_image)
# plt.imshow(comp_img)
# plt.show() 
# comp_img.save(r'C:/Users/CVPR/Soumi DI/NewTestSamples/New folder/7_comp0.jpg')
