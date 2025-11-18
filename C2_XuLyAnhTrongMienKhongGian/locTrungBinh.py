import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

# Đọc ảnh
img = cv2.imread("img/img1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Tạo mặt nạ lọc trung bình 7x7
k = np.ones((7, 7)) / (7 * 7)

# Tách các kênh màu
r, g, b = cv2.split(img)

# Lọc từng kênh bằng convolution 2D
R = convolve2d(r, k, mode='same', boundary='symm')
G = convolve2d(g, k, mode='same', boundary='symm')
B = convolve2d(b, k, mode='same', boundary='symm')

# Ghép lại ảnh sau khi lọc
imgC = cv2.merge((np.uint8(np.clip(R, 0, 255)),
                  np.uint8(np.clip(G, 0, 255)),
                  np.uint8(np.clip(B, 0, 255))))

# Hiển thị kết quả
plt.figure(dpi=600)
plt.subplot(1, 2, 1)
plt.axis("off")
plt.imshow(img)
plt.title("Ảnh gốc", fontsize=5)

plt.subplot(1, 2, 2)
plt.axis("off")
plt.imshow(imgC)
plt.title("Ảnh sau khi lọc trung bình", fontsize=5)

plt.show()
