import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sci

I= cv2.imread("img/circuit.tif", cv2.IMREAD_GRAYSCALE)
img = np.array(I, 'float')
kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
ky = np.flip(kx.T, axis=0)
imGx = sci.convolve2d(img, kx, mode='same')
imGy = sci.convolve2d(img, ky, mode='same')
gradM = np.sqrt(np.square(imGx) + np.square(imGy))
gradM = np.array(gradM*255/gradM.max(), 'uint8')
plt.figure(dpi = 300)
plt.subplot(1,4,1)
plt.axis("off")
plt.imshow(img,cmap="gray")
plt.title("Input Image", fontsize=8)
plt.subplot(1,4,2), plt.axis("off")
plt.imshow(imGx,cmap="gray")
plt.title("X Gradient", fontsize=8)
plt.subplot(1,4,3), plt.axis("off")
plt.imshow(imGy,cmap="gray")
plt.title("Y Gradient", fontsize=8)
plt.subplot(1,4,4), plt.axis("off")
plt.imshow(gradM,cmap="gray")
plt.title("Sobel Gradient", fontsize=8)
plt.show()