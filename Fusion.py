import numpy as np
import cv2
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

a=r'D:\Soumi\Test Sample/Cars197_km0.png'
img1=Image.open(a)
img1 = img1.convert('L') 
img1= asarray(img1)
w,h=img1.shape

b=r'C:\Users\CVPR\Desktop\Test images/car9dm.png'
img2=Image.open(b).convert('L')
img2 = img2.resize((h,w),Image.ANTIALIAS)
img2= asarray(img2)
num_zero = (img2 == 0).sum() #black
num_one = (img2 > 0).sum() #white
print('DM value count',num_zero,num_one)
image2=Image.fromarray(img2)
# img2.show()
# img2.save(r'C:/Users/CVPR/Soumi DI/NewTestSamples/New folder/7_DM6.jpg')
  

newimg=np.bitwise_or(img1,img2)
num_zeros = (newimg == 0).sum() #black
num_ones = (newimg > 0).sum() #white
print('Fusion value count',num_zeros,num_ones)
# print(newimg)
pil_image=Image.fromarray(newimg)
pil_image.show()
# pil_image.save(r'C:/Users/CVPR/Soumi DI/NewTestSamples/New folder/4_fusion.jpg')


OriginalImg=r'C:\Users\CVPR\Desktop\Test images/car9.jpg'
OriginalImg=Image.open(OriginalImg).convert('RGB')
print('Original image Size',OriginalImg.size)
print('DM image size',w,h)
# npOriginalImg = OriginalImg.convert('L') 
# FusionImage=np.bitwise_and(img2,npOriginalImg)  
# FusionImage=Image.fromarray(FusionImage)
blank = OriginalImg.point(lambda _: 0)
comp=Image.composite(OriginalImg,blank,pil_image)
# comp=Image.composite(OriginalImg,blank,image2)
plt.imshow(comp)
plt.show() 
comp.save(r'C:\Users\CVPR\Desktop\Test images/car9_comp.png')


# lp = comp.crop((131, 90, 165, 99))
# histog, bin_edges = np.histogram(lp, bins=256, range=(0, 256))
# plt.plot(histog) 
# plt.show()

# nlp = comp.crop((138, 43, 207, 80))
# histog, bin_edges = np.histogram(nlp, bins=256, range=(0, 256))
# plt.plot(histog) 
# plt.show()


dmwhite= np.argwhere(newimg>0)
print(dmwhite)
# grayimg=np.asarray(OriginalImg)
pixels=[]
# h,w=OriginalImg.size
for i in dmwhite:   
    coordinate=x,y=i[1],i[0]
    # print(coordinate)
    # print(OriginalImg.getpixel(coordinate))
    pixel_value=(OriginalImg.getpixel(coordinate))
    pixels.append(pixel_value)
    # pixels.append(OriginalImg.getpixel(coordinate))
    # print(pixels.appened(i))

# pixels=np.array(pixels)
# pixels=pixels/255.0
print("mean value",np.mean(pixels))
print("std value", np.std(pixels))

###############################    iteration    #####################################################
# i=0
# while (i<5):
#     inputimg=r'C:/Users/CVPR/Soumi DI/NewTestSamples/New folder/7_comp'+'i'+'.jpg'
