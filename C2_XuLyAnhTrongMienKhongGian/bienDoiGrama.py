import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img/hulk.webp")
# Chuyển đổi từ BGR sang RGB để hiển thị đúng màu với matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize = (12, 8), dpi = 100)
vals = [1, 0.1, 0.5, 1.5, 2.0, 2.5]
for i , gamma in enumerate(vals):
    c = 255 / (np.max(img_rgb) ** gamma)
    img_gamma = np.array(c * (img_rgb ** gamma), dtype = 'uint8')

    subf = plt.subplot(2, 3, i + 1)
    subf.set_title('gamma = ' + str(gamma), fontsize=10)
    subf.imshow(img_gamma)
    subf.axis('off')

plt.show()