import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img/img1.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

c = 255 / np.log(1 + np.max(img_gray))
img_log = c * np.log(1 + img_gray)
# Chuyển đổi về uint8 để hiển thị đúng
img_log = np.uint8(img_log)

plt.figure(dpi=300)
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(img_gray, cmap='gray')
plt.title("Ảnh mức xám gốc")

plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(img_log, cmap='gray')
plt.title("Ảnh sau biến đổi logarit")

plt.show()
