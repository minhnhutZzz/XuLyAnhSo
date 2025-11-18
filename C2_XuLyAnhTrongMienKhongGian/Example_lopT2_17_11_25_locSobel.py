import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sci

# a) đọc ảnh 
img_a = cv2.imread("img/body_bones_scan.jpg", 0)  
img_a = np.array(img_a, 'float')

# b) Áp dụng bộ lọc Laplacian
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])
img_b = sci.convolve2d(img_a, laplacian_kernel, mode='same')

# c) Cộng a,b
img_c = img_a + img_b
img_c = np.clip(img_c, 0, 255) # giới hạn gt

# d) Tính Gradient Sobel của a
kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])
ky = np.flip(kx.T, axis=0)
imGx = sci.convolve2d(img_a, kx, mode='same')
imGy = sci.convolve2d(img_a, ky, mode='same')
img_d = np.sqrt(np.square(imGx) + np.square(imGy))

# e)Làm trơn
box_filter_5x5 = np.ones((5, 5)) / 25
img_e = sci.convolve2d(img_d, box_filter_5x5, mode='same')

#  Ảnh mặt nạ = tích của (b) và (e)
# Chuẩn hóa ve [0,1]
img_b_norm = (img_b - img_b.min()) / (img_b.max() - img_b.min() + 1e-10)
img_e_norm = (img_e - img_e.min()) / (img_e.max() - img_e.min() + 1e-10)
img_f = img_b_norm * img_e_norm

# (g) Ảnh làm sắc nét = a + f
# Chuyển đổi mặt nạ về phạm vi phù hợp
img_f_scaled = img_f * 255 # chuyển đổi ve [0,255]
img_g = img_a + img_f_scaled
img_g = np.clip(img_g, 0, 255) # giới hạn gt

# (h) Biến đổi lũy thừa (hiệu chỉnh gamma)
gamma = 0.5  
img_g_norm = img_g / 255.0
img_h = np.power(img_g_norm, gamma) * 255
img_h = np.clip(img_h, 0, 255) # giới hạn gt

# Hiển thị kết quả
plt.figure(dpi=150, figsize=(10, 8))

# Hàng 1: (a), (b), (c), (d)
plt.subplot(2, 4, 1)
plt.imshow(img_a, cmap="gray")
plt.axis("off")
plt.title("(a) Ảnh gốc", fontsize=5)

plt.subplot(2, 4, 2)
plt.imshow(img_b, cmap="gray")
plt.axis("off")
plt.title("(b) Laplacian", fontsize=5)

plt.subplot(2, 4, 3)
plt.imshow(img_c, cmap="gray")
plt.axis("off")
plt.title("(c) Làm sắc nét", fontsize=5)

plt.subplot(2, 4, 4)
plt.imshow(img_d, cmap="gray")
plt.axis("off")
plt.title("(d) Gradient Sobel", fontsize=5)

# Hàng 2: (e), (f), (g), (h)
plt.subplot(2, 4, 5)
plt.imshow(img_e, cmap="gray")
plt.axis("off")
plt.title("(e) Sobel d", fontsize=5)

plt.subplot(2, 4, 6)
plt.imshow(img_f, cmap="gray")
plt.axis("off")
plt.title("(f) Mặt nạ b*e", fontsize=5)

plt.subplot(2, 4, 7)
plt.imshow(img_g, cmap="gray")
plt.axis("off")
plt.title("(g) Làm sắc nét a+f", fontsize=5)

plt.subplot(2, 4, 8)    
plt.imshow(img_h, cmap="gray")
plt.axis("off")
plt.title("(h) Biến đổi lũy grama ", fontsize=5)

plt.tight_layout()
plt.show()

