import cv2
import matplotlib.pyplot as plt
import numpy as np

# Đọc ảnh xám
img = cv2.imread("img/img1.jpg", cv2.IMREAD_GRAYSCALE)

# Cân bằng histogram
equ = cv2.equalizeHist(img)

# Hiển thị ảnh và histogram
plt.figure(dpi=600)

# Ảnh gốc
plt.subplot(2, 2, 1)
plt.axis("off")
plt.imshow(img, cmap='gray')
plt.title('Original Image')

# Histogram ảnh gốc
hist, bins = np.histogram(img.ravel(), 256, [0, 256])
plt.subplot(2, 2, 2)
plt.bar(bins[:-1], hist, color='0', width=1)
plt.title('Histogram (Original)')

# Ảnh sau cân bằng histogram
plt.subplot(2, 2, 3)
plt.axis("off")
plt.imshow(equ, cmap='gray')
plt.title('Equalized Image')

# Histogram sau cân bằng
hist, bins = np.histogram(equ.ravel(), 256, [0, 256])
plt.subplot(2, 2, 4)
plt.bar(bins[:-1], hist, color='0', width=1)
plt.title('Histogram (Equalized)')

plt.tight_layout()
plt.show()
