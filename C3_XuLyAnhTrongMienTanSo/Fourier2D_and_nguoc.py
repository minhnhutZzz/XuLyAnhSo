import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("img/dog.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
F = np.fft.fft2(img_gray)
fshift = np.fft.fftshift(F)
mag_spectrum =20*np.log(np.abs(fshift))
f = np.real(np.fft.ifft2(F))
fig =plt.figure(dpi = 300)
plt.subplot(1,3,1)
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("InputImage", fontsize=8)
plt.subplot(1,3,2)
plt.imshow(mag_spectrum, cmap="gray")
plt.axis("off")
plt.title("Spectrum", fontsize=8)
plt.subplot(1,3,3)
plt.imshow(f, cmap="gray")
plt.axis("off")
plt.title("IFFT image", fontsize=8)
plt.show()