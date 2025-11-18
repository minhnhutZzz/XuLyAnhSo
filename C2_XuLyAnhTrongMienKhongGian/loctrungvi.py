from skimage.util import random_noise
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img/img1.jpg', 0)
sp = random_noise(img, mode = 's&p', amount = 0.01)
h, w = img.shape
Ime = np.zeros((h, w))

for i in range(2, h):
    for j in range(2, w):
        Med = sp[i-1:i+1, j-1:j+1]
        Ime[i, j] = np.median(Med)

sp = np.array(sp * 255, dtype = np.uint8)
img1 = np.array(Ime * 255, dtype = np.uint8)

plt.figure(dpi = 600)
plt.subplot(1, 2, 1)
plt.axis('off')
plt.imshow(sp, cmap = plt.cm.gray)
plt.title('Noise image')

plt.subplot(1, 2, 2)
plt.axis('off')
plt.imshow(img1, cmap = plt.cm.gray)
plt.title('Median filter image')
