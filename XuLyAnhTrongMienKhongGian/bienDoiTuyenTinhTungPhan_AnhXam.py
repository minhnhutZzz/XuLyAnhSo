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

img = cv2.imread("img/hulk.webp")
# Chuyển đổi sang ảnh mức xám
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

a=100
b=150
v=70
w=190
pixel_vec = np.vectorize(PLTrans)
PLT= pixel_vec(img_gray, 255, a, b, v, w)
img_PTL = np.array(PLT, dtype = np.uint8)

plt.figure(dpi = 300)
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(img_gray, cmap = 'gray')
plt.title("Original gray image", fontsize = 10)
plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(img_PTL, cmap = 'gray')
plt.title('Piece-wise Linear transformed image', fontsize = 10)

plt.show()

