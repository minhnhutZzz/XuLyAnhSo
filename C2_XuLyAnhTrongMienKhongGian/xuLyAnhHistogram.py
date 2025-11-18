import cv2
import matplotlib.pyplot as plt
import numpy as np

img  = cv2.imread("img/bien640_480.jpg")
I = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8), dpi = 100)
plt.subplot(2,2,1)
plt.axis("off")
plt.imshow(I)
plt.title('Color image', fontsize=10)

Color = ['red', 'green', 'blue']
for i in range(3):
    # Tính histogram: hist = tần suất, bins = các mức cường độ (0-255)
    # hist: số lượng pixel có cường độ tương ứng (trục đứng - Y-axis)
    # bins: các giá trị cường độ từ 0-255 (trục ngang - X-axis)
    hist,bins = np.histogram(I[:,:,i].ravel(), 256, [0,255])
    plt.subplot(2,2,i+2)
    plt.bar(bins[0:-1], hist, color = Color[i], width=1)
    plt.title(Color[i] + ' histogram', fontsize=10)
    plt.xlabel('Intensity (0=tối, 255=sáng)', fontsize=8)
    plt.ylabel('Frequency (Số pixel)', fontsize=8)

plt.tight_layout()
plt.show()