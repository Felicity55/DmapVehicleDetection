from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
from skimage import color
from skimage import io
from PIL import Image
import numpy as np
import matplotlib.cm as CM

imgpath=r"D:\Soumi\Test Sample/Cars197.png"
img=cv2.imread(imgpath)
grayimg=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
vectorized = grayimg.reshape((-1,1))
vectorized = np.float32(vectorized)
K = 2
attempts=50
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((grayimg.shape))
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(result_image,kernel,iterations = 5)
# binimg=cv2.cvtColor(result_image,cv2.THRESH_BINARY)
plt.imshow(result_image, cmap=CM.gray)
# plt.title('Segmented Image when K = 2'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imsave("D:\Soumi\Test Sample/Cars197_km0.png",result_image, cmap=CM.gray)