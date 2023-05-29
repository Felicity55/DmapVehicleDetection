import numpy as np

from PIL import Image
from numpy import asarray
a=r'C:\Users\CVPR\Desktop\NewTestSamples\5_KM.jpg'
img1=Image.open(a)
img1 = img1.convert('L') 
img1= asarray(img1)
w,h=img1.shape
b=r'C:\Users\CVPR\Desktop\NewTestSamples\5_DM.jpg'
img2=Image.open(b)
img2 = img2.resize((h,w),Image.ANTIALIAS)
img2= asarray(img2)
pil_image=Image.fromarray(img2)
pil_image.show()
pil_image.save(r'C:\Users\CVPR\Desktop\NewTestSamples\5_DM.jpg')
  

# newimg=np.bitwise_or(img1,img2)
# print(newimg)
# pil_image=Image.fromarray(newimg)
# pil_image.show()
# pil_image.save(r'C:\Users\CVPR\Desktop\NewTestSamples\6_DM.jpg')
  
  