import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

img = cv2.imread("img/dog.jpg", 0)
img_copy = img.copy()
img = gaussian_filter(img, 3)
M, N = img.shape
pi = np.pi
C = -10
F = np.fft.fft2(img)
u = np.arange(0,M,1)
v = np.arange(0,N,1)
idx = (u > M/2)
u[idx] = u[idx] - M
idy = (v > N/2)
v[idy] = v[idy] - N
[V,U] = np.meshgrid(v,u)
D = np.sqrt(np.power(U, 2) + np.power(V, 2))
H = (-4 * pi * pi) * np.power(D, 2)
G = H * F
Out = np.real(np.fft.ifft2(G))
imgOut = Out / np.max(Out)
imfFilter = np.array(img + C * imgOut, 'uint8')
fig = plt.figure(dpi = 600)
plt.subplot(1,3,1)
plt.imshow(img_copy, cmap="gray")
plt.axis("off")
plt.title("InputImage", fontsize=3)
plt.subplot(1,3,2)
plt.imshow(Out, cmap="gray")
plt.axis("off")
plt.title("Laplacian HF image", fontsize=3)
plt.subplot(1,3,3)
plt.imshow(imfFilter, cmap="gray")
plt.axis("off")
plt.title("Image Enhenced using Laplacian HF", fontsize=3)
plt.show()