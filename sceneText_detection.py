import matplotlib.pyplot as plt
from PIL import Image
import cv2
import easyocr
imgpath=r"C:\Users\CVPR\Desktop\NewTestSamples/6_comp.jpg"
image= cv2.imread(imgpath)

# plt.imshow(image)
# plt.show()
x=2.0
y=2.0


# image = cv2.resize(image, (0, 0), fx=x, fy=y, interpolation = cv2.INTERSECT_FULL)
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
result = reader.readtext(image)

# print(result)
while result!=[]:
    x= x/2
    y= y/2
    #cv2.imwrite(r'C:\Users\CVPR\Desktop\NewTestSamples/5_comp2.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image = cv2.resize(image, (0, 0), fx=x, fy=y, interpolation = cv2.INTERSECT_FULL)
    img=image.copy()
    result = reader.readtext(image)
    for (bbox, text, prob) in result:
        print('Info: {} : {}'.format(text, prob))
        print(bbox,x)
        (xmin,ymin,xmax,ymax)=bbox
        xmin=(int(xmin[0]), int(xmin[1]))
        ymin=(int(ymin[0]), int(ymin[1]))
        xmax=(int(xmax[0]), int(xmax[1]))
        ymax=(int(ymax[0]), int(ymax[1]))
        
        # text= cleanup_text(text) 
        cv2.rectangle(img,xmin,xmax,(255,0,0),2)
        cv2.putText(img,text,(xmin[0],xmin[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
    plt.imshow(img)
    plt.show()
else:
     print(x)
    

    