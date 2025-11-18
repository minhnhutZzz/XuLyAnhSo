import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img/img9.jpg")
I = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

c = 255 / np.log(1 + np.max(I))
img_log = c * np.log(1 + I)
img_log = np.uint8(I)

plt.figure(dpi=300)
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(I)
plt.title("Ảnh mức xám gốc")

plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(img_log)
plt.title("Ảnh sau biến đổi logarit")

plt.show()
