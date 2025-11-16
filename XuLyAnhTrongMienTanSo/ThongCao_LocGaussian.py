import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img/img5.jpg", 0)
M, N = img.shape
F = np.fft.fft2(img)
u = np.arange(0, M)
v = np.arange(0, N)
idx = (u > M/2)
u[idx] = u[idx] - M
idy = (v > N/2)
v[idy] = v[idy] - N
[V, U] = np.meshgrid(v, u)

D0 = 25
D = np.sqrt(U**2 + V**2)
H = 1 - np.exp(-(D**2)/(2*(D0**2)))
G = H * F
imgOut = np.real(np.fft.ifft2(G))

plt.figure(dpi = 300)
plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("InputImage", fontsize=8)

plt.subplot(1,2,2)
plt.imshow(imgOut, cmap="gray")
plt.axis("off")
plt.title("Filtered Image", fontsize=8)
plt.show()











