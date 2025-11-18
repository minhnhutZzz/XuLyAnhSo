import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img/dog.jpg", 0)
F =np.fft.fft2(img)
M,N = img.shape
C0 = 90
W = 90
u = np.arange(0,M)
v = np.arange(0,N)
idx = (u > M/2)
u[idx] = u[idx] - M
idy = (v > N/2)
v[idy] = v[idy] - N
[V,U] = np.meshgrid(v,u)
D = np.sqrt(U**2 + V**2)
H = ((D>=(C0-W/2)) & (D<=(C0+W/2))).astype(float)
G = H * F
imgOut = np.real(np.fft.ifft2(G))

plt.figure(dpi = 300)
plt.subplot(2,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("InputImage", fontsize=3)

plt.subplot(2,2,2)
plt.imshow(np.fft.fftshift(H), cmap="gray")
plt.axis("off")
plt.title("Kernel H", fontsize=3)

plt.subplot(2,2,3)
plt.imshow(imgOut, cmap="gray")
plt.axis("off")
plt.title("Ideal BandPass Ly Tuong", fontsize=3)

plt.subplot(2,2,4)
plt.imshow(np.abs(imgOut), cmap="gray")
plt.axis("off")
plt.title("Ideal BandPass Ly Tuong (abs)", fontsize=3)

plt.show()