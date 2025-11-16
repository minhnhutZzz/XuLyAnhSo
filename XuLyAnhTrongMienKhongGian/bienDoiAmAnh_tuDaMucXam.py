import cv2
import matplotlib.pyplot as plt

img =  cv2.imread("img/hulk.webp")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_neg = 255 - img_GRAY
#img_neg = img.copy()
'''height, width = img.shape[:2]
for i in range(0, height):
    for j in range (0, width):
        img_neg[i,j] = 255 - img[i,j]'''

plt.figure(dpi = 300)
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(img_GRAY, cmap= plt.cm.gray)
plt.title("Ảnh gốc")

plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(img_neg, cmap= plt.cm.gray)
plt.title("Ảnh âm bản xám")

plt.imshow(img_neg, cmap="gray")
plt.show()