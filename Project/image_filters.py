import numpy as np


# Hàm Convolution 2D tùy chỉnh
def custom_conv2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(kernel * region)
    return output


# Hàm Áp dụng convolution
def Conv(img, k):
    Input = np.array(img, dtype=np.float32)
    if Input.ndim > 2:
        Out = np.zeros_like(Input)
        for i in range(3):
            Out[:,:,i] = custom_conv2d(Input[:,:,i], k)
    else:
        Out = custom_conv2d(Input, k)
    return np.clip(Out, 0, 255).astype(np.uint8)


# Hàm Tạo kernel Gaussian
def Gausskernel(l=5, sig=1.0):
    s = (l - 1) // 2
    ax = np.linspace(-s, s, l)
    gauss = np.exp( - (ax ** 2) / (2 * sig ** 2) )
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


# Hàm Áp dụng bộ lọc neighborhood
def apply_neighborhood_filter(img, size, filter_type):
    h, w, _ = img.shape
    pad = (size - 1) // 2
    padded_channels = [np.pad(img[:,:,c], pad, mode='constant', constant_values=0) for c in range(3)]
    out_channels = []
    for c in range(3):
        out_c = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                region = padded_channels[c][i:i+size, j:j+size]
                if filter_type == 'median':
                    val = int(np.median(region))
                elif filter_type == 'max':
                    val = int(np.max(region))
                elif filter_type == 'min':
                    val = int(np.min(region))
                elif filter_type == 'midpoint':
                    val = int( (np.max(region) + np.min(region)) / 2 )
                out_c[i,j] = val
        out_channels.append(out_c)
    return np.dstack(out_channels)

