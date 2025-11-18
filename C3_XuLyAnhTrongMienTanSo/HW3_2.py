import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img/PCB_xray_330_334.jpg", 0)
M, N = img.shape
F = np.fft.fft2(img)

u = np.arange(0, M)
v = np.arange(0, N)
idx = (u > M / 2)
u[idx] = u[idx] - M
idy = (v > N / 2)
v[idy] = v[idy] - N
[V, U] = np.meshgrid(v, u)

D0 = 30
D = np.sqrt(U**2 + V**2)
H = 1- np.exp(-(D**2) / (2 * (D0**2))).astype(float)
pass_list = [1, 10, 100]
filtered_imgs = []

for count in pass_list:
    img_temp = img.astype(np.float32)
    for _ in range(count):
        F_temp = np.fft.fft2(img_temp)
        G = H * F_temp
        img_temp = np.real(np.fft.ifft2(G))
    img_temp = cv2.normalize(img_temp, None, 0, 255, cv2.NORM_MINMAX)
    filtered_imgs.append(img_temp.astype(np.uint8))

plt.figure(dpi=300)
plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Input (330x334)", fontsize=5)

titles = ["Gaussian HighBass - 1lan", "Gaussian HighBass - 10lan", "Gaussian HighBass - 100lan"]
for idx, (title, img_out) in enumerate(zip(titles, filtered_imgs), start=2):
    plt.subplot(2, 2, idx)
    plt.imshow(img_out, cmap="gray")
    plt.axis("off")
    plt.title(title, fontsize=5)

plt.tight_layout()
plt.show()



