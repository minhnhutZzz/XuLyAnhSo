import numpy as np
import cv2
from PIL import Image


# Hàm Biến đổi âm bản
def apply_negative(image):
    img_rgb = np.array(image.convert('RGB'))
    img_neg = 255 - img_rgb
    return Image.fromarray(img_neg, mode='RGB')


# Hàm Biến đổi log
def apply_log_transform(image, c_factor=1.0):
    img_rgb = np.array(image.convert('RGB'), dtype=np.float32)
    
    # Tính hệ số c tối ưu tự động
    max_value = np.max(img_rgb)
    c_optimal = 255 / np.log(1 + max_value)
    
    # Áp dụng hệ số từ slider
    c = c_factor * c_optimal
    
    # Áp dụng công thức log: S = c * log(1 + r)
    img_log = c * np.log(1 + img_rgb)
    
    # Chuyển về uint8 và clip về [0, 255]
    img_log = np.clip(img_log, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_log, mode='RGB')


# Hàm Biến đổi tuyến tính từng phần
def apply_piecewise_linear(image, high_factor=1.0, low_factor=0.5):
    img_rgb = np.array(image.convert('RGB'), dtype=np.uint8)
    
    # Chuyển RGB sang HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    Ih, Is, Iv = cv2.split(hsv)
    
    # Lấy các tham số
    a = 50
    b = 150
    v = int(low_factor * 200)  # Scale từ 0.1-1.0 thành 20-200
    w = int(high_factor * 100)  # Scale từ 0.1-3.0 thành 10-300
    
    # Hàm piecewise linear transformation
    def PLTrans(val, L, a, b, v, w):
        if 0 <= val < a:
            return (v / a) * val
        elif a <= val < b:
            return ((w - v) / (b - a)) * (val - a) + v
        else:
            return ((L - w) / (L - b)) * (val - b) + w
    
    # Áp dụng transformation cho kênh V (Value) trong HSV
    pixel_vec = np.vectorize(PLTrans)
    PLT = pixel_vec(Iv, 255, a, b, v, w)
    img_PLT = np.array(PLT, dtype=np.uint8)
    
    # Merge lại thành HSV image
    hsv_image = cv2.merge([Ih, Is, img_PLT])
    
    # Chuyển HSV về RGB
    img_rgb_out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(img_rgb_out, mode='RGB')


# Hàm Biến đổi gamma
def apply_gamma_transform(image, gamma=1.0, c_factor=1.0):
    img_rgb = np.array(image.convert('RGB'), dtype=np.float32)
    
    # Tính hệ số c tối ưu tự động
    max_value = np.max(img_rgb)
    c_optimal = 255 / (max_value ** gamma)
    
    # Áp dụng hệ số từ slider
    c = c_factor * c_optimal
    
    # Áp dụng công thức gamma: S = c * (r^gamma)
    img_gamma = c * (img_rgb ** gamma)
    
    # Chuyển về uint8 và clip về [0, 255]
    img_gamma = np.clip(img_gamma, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_gamma, mode='RGB')


# Hàm Cân bằng histogram
def apply_histogram_equalization(image, brightness_factor=1.0):
    img_rgb = np.array(image.convert('RGB'), dtype=np.uint8)
    
    # Chuyển RGB sang BGR cho OpenCV
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Tách các kênh màu
    b, g, r = cv2.split(img_bgr)
    
    # Cân bằng histogram cho từng kênh
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    
    # Ghép lại các kênh
    equ_bgr = cv2.merge([b_eq, g_eq, r_eq])
    
    # Chuyển BGR về RGB
    equ_rgb = cv2.cvtColor(equ_bgr, cv2.COLOR_BGR2RGB)
    
    # Áp dụng hệ số điều chỉnh độ sáng
    if brightness_factor != 1.0:
        equ_rgb = np.clip(equ_rgb.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
    
    return Image.fromarray(equ_rgb, mode='RGB')

