import numpy as np
import cv2
from PIL import Image
import time


def _prepare_frequency_domain(img):
    """Chuẩn bị ảnh cho xử lý trong miền tần số"""
    # B1: Tiền xử lý ảnh - chuyển sang HSV và lấy kênh V
    start_b1 = time.time()
    if isinstance(img, Image.Image):
        img_rgb = np.array(img.convert('RGB'))
    else:
        img_rgb = np.array(img)
        if len(img_rgb.shape) == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    
    # Chuyển RGB sang HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(img_hsv)
    
    # Lưu lại H và S để merge sau
    img_array = V.astype(np.float32)
    time_b1 = time.time() - start_b1
    
    M, N = img_array.shape
    
    # B2: Biến đổi Fourier
    start_b2 = time.time()
    F = np.fft.fft2(img_array)
    time_b2 = time.time() - start_b2
    
    # Tạo meshgrid với tâm ở (0,0) để tính khoảng cách tần số
    u = np.arange(0, M)
    v = np.arange(0, N)
    idx = (u > M/2)
    u[idx] = u[idx] - M
    idy = (v > N/2)
    v[idy] = v[idy] - N
    [V_grid, U] = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V_grid**2)
    
    return F, D, M, N, H, S, time_b1, time_b2


def apply_ideal_lowpass_filter(image, D0=50):
    """Lọc thông thấp lý tưởng"""
    total_start = time.time()
    
    # B1 & B2: Tiền xử lý và FFT
    F, D, M, N, H_channel, S_channel, time_b1, time_b2 = _prepare_frequency_domain(image)
    
    # B3: Xây dựng bộ lọc
    start_b3 = time.time()
    H = (D <= D0).astype(float)
    time_b3 = time.time() - start_b3
    
    # B4: Lọc trong miền tần số
    start_b4 = time.time()
    G = H * F
    time_b4 = time.time() - start_b4
    
    # B5: Biến đổi Fourier ngược
    start_b5 = time.time()
    V_filtered = np.real(np.fft.ifft2(G))
    time_b5 = time.time() - start_b5
    
    # B6: Hậu xử lý - merge lại với H và S, chuyển về RGB
    start_b6 = time.time()
    V_filtered = np.clip(V_filtered, 0, 255).astype(np.uint8)
    img_hsv_out = cv2.merge([H_channel, S_channel, V_filtered])
    img_rgb_out = cv2.cvtColor(img_hsv_out, cv2.COLOR_HSV2RGB)
    result = Image.fromarray(img_rgb_out, mode='RGB')
    time_b6 = time.time() - start_b6
    
    total_time = time.time() - total_start
    
    # In thời gian (console)
    print(f"\n=== Lọc Thông Thấp Lý Tưởng (D0={D0}) ===")
    print(f"B1 - Tiền xử lý:        {time_b1*1000:.3f} ms")
    print(f"B2 - FFT:               {time_b2*1000:.3f} ms")
    print(f"B3 - Xây dựng bộ lọc:   {time_b3*1000:.3f} ms")
    print(f"B4 - Lọc (F × H):       {time_b4*1000:.3f} ms")
    print(f"B5 - IFFT:              {time_b5*1000:.3f} ms")
    print(f"B6 - Hậu xử lý:         {time_b6*1000:.3f} ms")
    print(f"Tổng thời gian:         {total_time*1000:.3f} ms")
    print("=" * 40)
    
    timing_info = {
        'filter_name': 'Lọc Thông Thấp Lý Tưởng',
        'params': f'D0={D0}',
        'time_b1': time_b1 * 1000,
        'time_b2': time_b2 * 1000,
        'time_b3': time_b3 * 1000,
        'time_b4': time_b4 * 1000,
        'time_b5': time_b5 * 1000,
        'time_b6': time_b6 * 1000,
        'total_time': total_time * 1000
    }
    
    return result, timing_info


def apply_gaussian_lowpass_filter(image, D0=50):
    """Lọc thông thấp Gaussian"""
    total_start = time.time()
    
    # B1 & B2: Tiền xử lý và FFT
    F, D, M, N, H_channel, S_channel, time_b1, time_b2 = _prepare_frequency_domain(image)
    
    # B3: Xây dựng bộ lọc
    start_b3 = time.time()
    H = np.exp(-(D**2)/(2*(D0**2)))
    time_b3 = time.time() - start_b3
    
    # B4: Lọc trong miền tần số
    start_b4 = time.time()
    G = H * F
    time_b4 = time.time() - start_b4
    
    # B5: Biến đổi Fourier ngược
    start_b5 = time.time()
    V_filtered = np.real(np.fft.ifft2(G))
    time_b5 = time.time() - start_b5
    
    # B6: Hậu xử lý - merge lại với H và S, chuyển về RGB
    start_b6 = time.time()
    V_filtered = np.clip(V_filtered, 0, 255).astype(np.uint8)
    img_hsv_out = cv2.merge([H_channel, S_channel, V_filtered])
    img_rgb_out = cv2.cvtColor(img_hsv_out, cv2.COLOR_HSV2RGB)
    result = Image.fromarray(img_rgb_out, mode='RGB')
    time_b6 = time.time() - start_b6
    
    total_time = time.time() - total_start
    
    # In thời gian (console)
    print(f"\n=== Lọc Thông Thấp Gaussian (D0={D0}) ===")
    print(f"B1 - Tiền xử lý:        {time_b1*1000:.3f} ms")
    print(f"B2 - FFT:               {time_b2*1000:.3f} ms")
    print(f"B3 - Xây dựng bộ lọc:   {time_b3*1000:.3f} ms")
    print(f"B4 - Lọc (F × H):       {time_b4*1000:.3f} ms")
    print(f"B5 - IFFT:              {time_b5*1000:.3f} ms")
    print(f"B6 - Hậu xử lý:         {time_b6*1000:.3f} ms")
    print(f"Tổng thời gian:         {total_time*1000:.3f} ms")
    print("=" * 40)
    
    timing_info = {
        'filter_name': 'Lọc Thông Thấp Gaussian',
        'params': f'D0={D0}',
        'time_b1': time_b1 * 1000,
        'time_b2': time_b2 * 1000,
        'time_b3': time_b3 * 1000,
        'time_b4': time_b4 * 1000,
        'time_b5': time_b5 * 1000,
        'time_b6': time_b6 * 1000,
        'total_time': total_time * 1000
    }
    
    return result, timing_info


def apply_butterworth_lowpass_filter(image, D0=75, n=2):
    """Lọc thông thấp Butterworth"""
    total_start = time.time()
    
    # B1 & B2: Tiền xử lý và FFT
    F, D, M, N, H_channel, S_channel, time_b1, time_b2 = _prepare_frequency_domain(image)
    
    # B3: Xây dựng bộ lọc
    start_b3 = time.time()
    H = 1/(1+(D/D0)**(2*n))
    time_b3 = time.time() - start_b3
    
    # B4: Lọc trong miền tần số
    start_b4 = time.time()
    G = H * F
    time_b4 = time.time() - start_b4
    
    # B5: Biến đổi Fourier ngược
    start_b5 = time.time()
    V_filtered = np.real(np.fft.ifft2(G))
    time_b5 = time.time() - start_b5
    
    # B6: Hậu xử lý - merge lại với H và S, chuyển về RGB
    start_b6 = time.time()
    V_filtered = np.clip(V_filtered, 0, 255).astype(np.uint8)
    img_hsv_out = cv2.merge([H_channel, S_channel, V_filtered])
    img_rgb_out = cv2.cvtColor(img_hsv_out, cv2.COLOR_HSV2RGB)
    result = Image.fromarray(img_rgb_out, mode='RGB')
    time_b6 = time.time() - start_b6
    
    total_time = time.time() - total_start
    
    # In thời gian (console)
    print(f"\n=== Lọc Thông Thấp Butterworth (D0={D0}, n={n}) ===")
    print(f"B1 - Tiền xử lý:        {time_b1*1000:.3f} ms")
    print(f"B2 - FFT:               {time_b2*1000:.3f} ms")
    print(f"B3 - Xây dựng bộ lọc:   {time_b3*1000:.3f} ms")
    print(f"B4 - Lọc (F × H):       {time_b4*1000:.3f} ms")
    print(f"B5 - IFFT:              {time_b5*1000:.3f} ms")
    print(f"B6 - Hậu xử lý:         {time_b6*1000:.3f} ms")
    print(f"Tổng thời gian:         {total_time*1000:.3f} ms")
    print("=" * 40)
    
    timing_info = {
        'filter_name': 'Lọc Thông Thấp Butterworth',
        'params': f'D0={D0}, n={n}',
        'time_b1': time_b1 * 1000,
        'time_b2': time_b2 * 1000,
        'time_b3': time_b3 * 1000,
        'time_b4': time_b4 * 1000,
        'time_b5': time_b5 * 1000,
        'time_b6': time_b6 * 1000,
        'total_time': total_time * 1000
    }
    
    return result, timing_info


def apply_ideal_highpass_filter(image, D0=10):
    """Lọc thông cao lý tưởng"""
    total_start = time.time()
    
    # B1 & B2: Tiền xử lý và FFT
    F, D, M, N, H_channel, S_channel, time_b1, time_b2 = _prepare_frequency_domain(image)
    
    # B3: Xây dựng bộ lọc
    start_b3 = time.time()
    H = (D >= D0).astype(float)
    time_b3 = time.time() - start_b3
    
    # B4: Lọc trong miền tần số
    start_b4 = time.time()
    G = H * F
    time_b4 = time.time() - start_b4
    
    # B5: Biến đổi Fourier ngược
    start_b5 = time.time()
    V_filtered = np.real(np.fft.ifft2(G))
    time_b5 = time.time() - start_b5
    
    # B6: Hậu xử lý - merge lại với H và S, chuyển về RGB
    start_b6 = time.time()
    V_filtered = np.clip(V_filtered, 0, 255).astype(np.uint8)
    img_hsv_out = cv2.merge([H_channel, S_channel, V_filtered])
    img_rgb_out = cv2.cvtColor(img_hsv_out, cv2.COLOR_HSV2RGB)
    result = Image.fromarray(img_rgb_out, mode='RGB')
    time_b6 = time.time() - start_b6
    
    total_time = time.time() - total_start
    
    # In thời gian (console)
    print(f"\n=== Lọc Thông Cao Lý Tưởng (D0={D0}) ===")
    print(f"B1 - Tiền xử lý:        {time_b1*1000:.3f} ms")
    print(f"B2 - FFT:               {time_b2*1000:.3f} ms")
    print(f"B3 - Xây dựng bộ lọc:   {time_b3*1000:.3f} ms")
    print(f"B4 - Lọc (F × H):       {time_b4*1000:.3f} ms")
    print(f"B5 - IFFT:              {time_b5*1000:.3f} ms")
    print(f"B6 - Hậu xử lý:         {time_b6*1000:.3f} ms")
    print(f"Tổng thời gian:         {total_time*1000:.3f} ms")
    print("=" * 40)
    
    timing_info = {
        'filter_name': 'Lọc Thông Cao Lý Tưởng',
        'params': f'D0={D0}',
        'time_b1': time_b1 * 1000,
        'time_b2': time_b2 * 1000,
        'time_b3': time_b3 * 1000,
        'time_b4': time_b4 * 1000,
        'time_b5': time_b5 * 1000,
        'time_b6': time_b6 * 1000,
        'total_time': total_time * 1000
    }
    
    return result, timing_info


def apply_gaussian_highpass_filter(image, D0=25):
    """Lọc thông cao Gaussian"""
    total_start = time.time()
    
    # B1 & B2: Tiền xử lý và FFT
    F, D, M, N, H_channel, S_channel, time_b1, time_b2 = _prepare_frequency_domain(image)
    
    # B3: Xây dựng bộ lọc
    start_b3 = time.time()
    H = 1 - np.exp(-(D**2)/(2*(D0**2)))
    time_b3 = time.time() - start_b3
    
    # B4: Lọc trong miền tần số
    start_b4 = time.time()
    G = H * F
    time_b4 = time.time() - start_b4
    
    # B5: Biến đổi Fourier ngược
    start_b5 = time.time()
    V_filtered = np.real(np.fft.ifft2(G))
    time_b5 = time.time() - start_b5
    
    # B6: Hậu xử lý - merge lại với H và S, chuyển về RGB
    start_b6 = time.time()
    V_filtered = np.clip(V_filtered, 0, 255).astype(np.uint8)
    img_hsv_out = cv2.merge([H_channel, S_channel, V_filtered])
    img_rgb_out = cv2.cvtColor(img_hsv_out, cv2.COLOR_HSV2RGB)
    result = Image.fromarray(img_rgb_out, mode='RGB')
    time_b6 = time.time() - start_b6
    
    total_time = time.time() - total_start
    
    # In thời gian (console)
    print(f"\n=== Lọc Thông Cao Gaussian (D0={D0}) ===")
    print(f"B1 - Tiền xử lý:        {time_b1*1000:.3f} ms")
    print(f"B2 - FFT:               {time_b2*1000:.3f} ms")
    print(f"B3 - Xây dựng bộ lọc:   {time_b3*1000:.3f} ms")
    print(f"B4 - Lọc (F × H):       {time_b4*1000:.3f} ms")
    print(f"B5 - IFFT:              {time_b5*1000:.3f} ms")
    print(f"B6 - Hậu xử lý:         {time_b6*1000:.3f} ms")
    print(f"Tổng thời gian:         {total_time*1000:.3f} ms")
    print("=" * 40)
    
    timing_info = {
        'filter_name': 'Lọc Thông Cao Gaussian',
        'params': f'D0={D0}',
        'time_b1': time_b1 * 1000,
        'time_b2': time_b2 * 1000,
        'time_b3': time_b3 * 1000,
        'time_b4': time_b4 * 1000,
        'time_b5': time_b5 * 1000,
        'time_b6': time_b6 * 1000,
        'total_time': total_time * 1000
    }
    
    return result, timing_info


def apply_butterworth_highpass_filter(image, D0=20, n=2):
    """Lọc thông cao Butterworth"""
    total_start = time.time()
    
    # B1 & B2: Tiền xử lý và FFT
    F, D, M, N, H_channel, S_channel, time_b1, time_b2 = _prepare_frequency_domain(image)
    
    # B3: Xây dựng bộ lọc
    start_b3 = time.time()
    H = 1/(1+(D0/D)**(2*n))
    time_b3 = time.time() - start_b3
    
    # B4: Lọc trong miền tần số
    start_b4 = time.time()
    G = H * F
    time_b4 = time.time() - start_b4
    
    # B5: Biến đổi Fourier ngược
    start_b5 = time.time()
    V_filtered = np.real(np.fft.ifft2(G))
    time_b5 = time.time() - start_b5
    
    # B6: Hậu xử lý - merge lại với H và S, chuyển về RGB
    start_b6 = time.time()
    V_filtered = np.clip(V_filtered, 0, 255).astype(np.uint8)
    img_hsv_out = cv2.merge([H_channel, S_channel, V_filtered])
    img_rgb_out = cv2.cvtColor(img_hsv_out, cv2.COLOR_HSV2RGB)
    result = Image.fromarray(img_rgb_out, mode='RGB')
    time_b6 = time.time() - start_b6
    
    total_time = time.time() - total_start
    
    # In thời gian (console)
    print(f"\n=== Lọc Thông Cao Butterworth (D0={D0}, n={n}) ===")
    print(f"B1 - Tiền xử lý:        {time_b1*1000:.3f} ms")
    print(f"B2 - FFT:               {time_b2*1000:.3f} ms")
    print(f"B3 - Xây dựng bộ lọc:   {time_b3*1000:.3f} ms")
    print(f"B4 - Lọc (F × H):       {time_b4*1000:.3f} ms")
    print(f"B5 - IFFT:              {time_b5*1000:.3f} ms")
    print(f"B6 - Hậu xử lý:         {time_b6*1000:.3f} ms")
    print(f"Tổng thời gian:         {total_time*1000:.3f} ms")
    print("=" * 40)
    
    timing_info = {
        'filter_name': 'Lọc Thông Cao Butterworth',
        'params': f'D0={D0}, n={n}',
        'time_b1': time_b1 * 1000,
        'time_b2': time_b2 * 1000,
        'time_b3': time_b3 * 1000,
        'time_b4': time_b4 * 1000,
        'time_b5': time_b5 * 1000,
        'time_b6': time_b6 * 1000,
        'total_time': total_time * 1000
    }
    
    return result, timing_info

