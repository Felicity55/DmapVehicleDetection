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
import skimage.io
import skimage.color
import skimage.filters
from skimage.segmentation import chan_vese
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.morphology import erosion, dilation
# Global variables
use_gpu = torch.cuda.is_available()
checkpoint_path = r'C:\Users\CVPR\source\repos\MCNN-pytorch-master\checkpoints\epoch_49.param'
device = torch.device('cuda:0' if use_gpu else 'cpu')


def infer(sample):
    model = MCNN().to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'])
 
    
    img = sample
    # density_map = dm
    # gt_count = sample['gt_count']
    
    # make img's dimension (1, C, H, W)
    img = torch.unsqueeze(img, dim=0)
    # make the dimension of the density map (1, H, W)
    # density_map = torch.unsqueeze(density_map, dim=0)
    
    img = img.to(device)
    # density_map = density_map.to(device)
    
    with torch.set_grad_enabled(False):
        et_dm = model(img).detach()
        # down_sample = nn.Sequential(nn.MaxPool2d(2), nn.MaxPool2d(2))
        # down_gt_dm = down_sample(density_map)

        et_dm = et_dm.squeeze(0).cpu().numpy()
        # down_gt_dm = down_gt_dm.cpu().numpy()
        # et_dm_reshape = et_dm.copy().reshape(-1, 1)
    print(f'The integral of the estimated density map: {et_dm.sum()}')
    print(et_dm.shape)
    print(type(et_dm))
   
    x=et_dm.squeeze()
   
    plt.imshow(x,cmap=CM.jet)
    # plt.pause(0.001)
    plt.show()
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


    return torch.squeeze(img, dim=0).cpu().numpy(), et_dm 
    # , density_map
    # , gt_count

data_trans = transforms.Compose([transforms.ToTensor()])
imgpath=r'C:\Users\CVPR\source\repos\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\test_data\images\IMG_1.jpg'
# imgpath=r'C:\Users\CVPR\Desktop\CrowdVehicles\p5.jpg'
# imgpath=r'C:\Users\CVPR\Desktop\846-20160429-01\00001.jpg'
# dmpath=r"C:\Users\CVPR\source\repos\Detection\CityCam\846\846-20160429-07\000001_dm.npy"
img=cv2.imread(imgpath)
img= cv2. cvtColor(img, cv2.COLOR_BGR2RGB)
# img=img.astype('uint8')
print(img.shape)
print(type(img))
img=data_trans(img)
print(img.shape)
# dmap=np.load(dmpath)
# dmap=data_trans(dmap)
infer(img)