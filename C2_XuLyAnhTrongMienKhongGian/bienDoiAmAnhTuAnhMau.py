import cv2
import matplotlib.pyplot as plt

img =  cv2.imread("img/hulk.webp")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_neg = 255 - img_rgb

plt.figure(dpi = 300)
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(img_rgb)
plt.title("Ảnh gốc")

plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(img_neg, cmap= plt.cm.gray)
plt.title("Ảnh âm bản màu")
plt.imshow(cv2.cvtColor(img_neg, cv2.COLOR_BGR2RGB))
plt.show()