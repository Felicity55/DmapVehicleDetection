from cv2 import countNonZero
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
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
import os
import cv2
sys.path.append('..')

from PIL import Image
import scipy.spatial
import scipy.ndimage
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

from scipy.ndimage import gaussian_filter

# Global variables
use_gpu = torch.cuda.is_available()
checkpoint_path = r'D:\Soumi\Soumi DI\DmapVehicleDetection\checkpoints\CSRNet-Epochs-5000_BatchSize-8_LR-0.0001_Momentum-0.95_Gamma-0.5_Version-1\best_model.pt'
device = torch.device('cuda:0' if use_gpu else 'cpu')


def infer(sample):
    model = MCNN()
    # checkpoint = torch.load(checkpoint_path) #28.9.22
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) #28.9.22// run with only cpu
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    img = sample
    # print("Image shape:",img.shape)
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
        # print("Estimated density map is:",et_dm)
        # print("Max value of Estimated Density Map is:",et_dm.max())
        # print("Min value of Estimated Density Map is:",et_dm.min())
        # # down_gt_dm = down_gt_dm.cpu().numpy()
        # et_dm_reshape = et_dm.copy().reshape(-1, 1)
    print(f'The integral of the estimated density map: {et_dm.sum()}')
    
    # print(et_dm.dtype)
   
    x=et_dm.squeeze()
    x = gaussian_filter(x, sigma=2)
    
    plt.imshow(x,cmap=CM.jet)
    # # plt.pause(0.001)
    # plt.show()

    x1d = x.copy().reshape(-1, 1)
    print("Values:",x1d)

    #K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=0, max_iter= 20).fit(x1d)
    y_pred=kmeans.predict(x1d)

    labels=kmeans.labels_
    centroids = kmeans.cluster_centers_
    print("Kmeans labels:",kmeans.labels_)
    # zeros=labels.count(0)
    num_zeros = (labels == 0)
    num_ones = (labels > 0)
    print(num_zeros.sum(),num_ones.sum())
    center= np.array(kmeans.cluster_centers_)
    
    zlevel_mean=np.mean(num_zeros)
    nzlevel_mean=np.mean(num_ones)
    print("Mean values:", zlevel_mean, nzlevel_mean)
    zlevel_std=np.std(num_zeros)
    nzlevel_std=np.std(num_ones)
    print("Standard Deviations:", zlevel_std, nzlevel_std)
    # print("centers:", center)
    # img=img.squeeze()
    width, height = x.shape
    y_pred = y_pred.reshape(width, height)
# print('y_pred shape ', y_pred.shape)
# print("y prediction",y_pred)
    a2 = centroids[labels]
# print('a2 shape ', a2.shape)
    a3 = a2.reshape(width, height)
    bin_img = np.where(a3== np.min(a3), 255, 0)
    num_zero = (bin_img == 0).sum() #black
    num_one = (bin_img != 0).sum() #white
    print(num_zero,num_one)
    bin_img=1-bin_img
    # print('a3 shape ', a3.shape)
    # print('a3 values ', a3)
    dmimage=bin_img*255
    # dmimage=Image.fromarray(bin_img*255)
    # dmimage=dmimage.resize((w,h))
    # plt.imshow(dmimage)
    # plt.show()
    # cv2.imwrite(r"C:\Users\CVPR\Desktop\demo\7_dm.png",dmimage)


    # print(f'The integral of the GT count: {gt_count}')
    # print(f'The integral of the GT density map {density_map.sum()}')
    # np.save((r"C:\Users\CVPR\source\repos\Detection\CityCam\segment\00006" + '_dm'+'.npy'), et_dm)
   

    # dm = scipy.ndimage.gaussian_filter(x, sigma=1)
    # # dm = density_map_generator((240, 352),x)
    # plt.imshow(dm)
    # plt.pause(0.001)
    # plt.show()
#     newimg=et_dm.transpose(1, 2, 0)

# # fit scaler on training data
#     scale=MinMaxScaler(feature_range=(0,255))
#     oned=newimg.reshape([-1,1])
#     norm=scale.fit_transform(oned)
#     norm=norm.reshape(newimg.shape)
#     # norm=norm.astype(int)
#     print(norm.shape)
#     print(norm.max())
    # ret, thresh = cv2.threshold(norm,0,255,cv2.THRESH_OTSU)
   
    # plt.figure(figsize=(8,8))
    # chanvese = chan_vese(binary_mask,max_iter=100,extended_output=True)
    # segment = slic(norm,n_segments=2,compactness=10)

   
    # kernel = np.ones((2,2),np.uint8)
    # dilation = cv2.dilate(x,kernel,iterations = 20)
    # t = skimage.filters.threshold_otsu(dilation)
    # binary_mask = dilation > t
    # plt.imshow(binary_mask,cmap='gray')
    # # plt.axis('off')
    # # plt.title("Threshold Image")
    # plt.show()


    # return torch.squeeze(img, dim=0).cpu().numpy(), et_dm 
    return dmimage
    # , density_map
    # , gt_count

data_trans = transforms.Compose([transforms.ToTensor()])
root='D:/Soumi/License plate detection/LP-night/3/3/test'
entries = os.listdir(root)

for file in entries:
    # Check whether file is in text format or not
    center_points = []
    coordinates=[]
    if  file.endswith(".jpg") :
        # file_path = f"{entries}\{file}"
        imgpath=os.path.join(root,file)

        # imgpath=r'C:\Users\CVPR\Desktop\CrowdVehicles\7.jpg'
        # imgpath=r'C:\Users\CVPR\Desktop\demo\7.jpg'
        # imgpath=r'C:\Users\CVPR\source\repos\Detection\CityCam\253\253-20160421-15\000001.jpg'
        # dmpath=r"C:\Users\CVPR\source\repos\Detection\CityCam\846\846-20160429-07\000001_dm.npy"
        image1=cv2.imread(imgpath)
        image= cv2. cvtColor(image1, cv2.COLOR_BGR2RGB)
        # img = cv2.GaussianBlur(img,(5,5),0)
        # img=img.astype('uint8')
        # print(img.shape)
        # print(type(img))


        img=data_trans(image)
        print("image shape is:",image.shape)
        h,w,c=image.shape
        # dmap=np.load(dmpath)
        # dmap=data_trans(dmap)
        dm=infer(img)
        image1=cv2.resize(image1,(int(w/4),int(h/4)), interpolation = cv2.INTER_AREA)
        # dmimage=Image.fromarray(dm*255)
        dmimage=image.resize((w,h))
        # print("type of dm is",type(dmimage))
        plt.imshow(image1)
        plt.imshow(dm, alpha=0.5)
        plt.show()

