import cv2
import matplotlib.pyplot as plt
import numpy as np

def PLTrans (val, L, a, b, v, w):
    if (0 <=val and val < a):
        return (v/a)*val
    elif ((a<= val) and (val < b)):
        return ((w - v)/(b - a))*(val - a) + v
    else:
        return ((L - w)/(L - b))*(val - b) + w

img = cv2.imread("img/sieuanhhung.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
Ih, Is, Iv = cv2.split(hsv)
a=50
b=150
v=100
w=220
pixel_vec = np.vectorize(PLTrans)
PLT = pixel_vec(Iv, 255, a, b, v, w)    
img_PLT = np.array(PLT, dtype = np.uint8)
hsv_image = cv2.merge([Ih, Is, img_PLT])
Out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
plt.figure(dpi = 300)
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(img)
plt.title('Original color image', fontsize = 8)
plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(Out)
plt.title('Piece-wise Linear transformed image', fontsize = 8)
plt.show()
