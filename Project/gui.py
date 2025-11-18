import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time

from image_filters import Conv, Gausskernel, apply_neighborhood_filter
from image_transforms import (
    apply_negative,
    apply_log_transform,
    apply_piecewise_linear,
    apply_gamma_transform,
    apply_histogram_equalization
)
from frequency_domain_filters import (
    apply_ideal_lowpass_filter,
    apply_gaussian_lowpass_filter,
    apply_butterworth_lowpass_filter,
    apply_ideal_highpass_filter,
    apply_gaussian_highpass_filter,
    apply_butterworth_highpass_filter
)


class ImageProcessorGUI:
    # Khởi tạo giao diện
    def __init__(self, root):
        self.root = root
        self.root.title("Phần Mềm Xử Lý Ảnh")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f2f5')
        
        # Biến lưu trữ ảnh
        self.original_image = None
        self.current_image = None
        self.photo_original = None
        self.photo_current = None
        # Biến để theo dõi chức năng đang được sử dụng
        self.active_transformation = None
        # Biến để theo dõi miền đang chọn (0: không gian, 1: tần số)
        self.domain_var = tk.IntVar(value=0)
        
        # Biến cho threading và debouncing
        self.processing_thread = None
        self.is_processing = False
        self.debounce_timer = None
        self.debounce_delay = 0.2  # 200ms debounce delay cho slider (tăng để tối ưu hiệu năng)
        
        # Thiết lập style cho giao diện sáng với màu xanh dương nổi bật
        self._setup_styles()
        self.setup_ui()
    
    # Thiết lập style cho giao diện
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        # Màu xanh dương chủ đạo
        self.blue_color = '#007bff'  # Primary blue
        self.blue_dark = '#0056b3'   # Darker blue
        self.blue_light = '#17a2b8'  # Lighter blue for accents
        self.gray_light = '#e9ecef'  # Light gray
        
        style.configure('TButton', padding=10, font=('Segoe UI', 10), background='#ffffff', foreground='#212529', borderwidth=1)
        style.map('TButton', 
                  background=[('active', self.gray_light), ('pressed', '#dee2e6')],
                  bordercolor=[('active', self.blue_color), ('focus', self.blue_color)])
        style.configure('TLabel', font=('Segoe UI', 10), background='#f0f2f5', foreground='#212529')
        style.configure('TFrame', background='#f0f2f5')
        style.configure('TLabelFrame', background='#f0f2f5', foreground='#212529', borderwidth=1, relief='solid')
        style.configure('TLabelFrame.Label', background='#f0f2f5', foreground='#212529', font=('Segoe UI', 10, 'bold'))
        style.configure('TCheckbutton', background='#f0f2f5', foreground='#212529', font=('Segoe UI', 10))
        style.map('TCheckbutton', 
                  background=[('active', self.gray_light)],
                  indicatorcolor=[('selected', self.blue_color)])
        style.configure('TScale', background='#f0f2f5', troughcolor=self.gray_light, borderwidth=0, foreground=self.blue_color, font=('Segoe UI', 9))
        style.map('TScale', 
                  background=[('active', '#f0f2f5')],
                  troughcolor=[('active', '#ced4da')])
    
    # Thiết lập giao diện người dùng
    def setup_ui(self):
        # Frame chính
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Frame bên trái: Hiển thị ảnh
        self._setup_left_frame(main_frame)
        
        # Frame bên phải: Chức năng chi tiết
        self._setup_right_frame(main_frame)
    
    # Thiết lập frame bên trái (hiển thị ảnh)
    def _setup_left_frame(self, parent):
        left_outer = tk.Frame(parent, bg=self.blue_color, highlightthickness=3, highlightbackground=self.blue_color, relief='solid')
        left_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        left_frame = tk.Frame(left_outer, bg='#ffffff')
        left_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tiêu đề
        title_frame = tk.Frame(left_frame, bg='#ffffff', height=50)
        title_frame.pack(fill=tk.X, pady=(5, 10))
        title_frame.pack_propagate(False)
        title_label = tk.Label(title_frame, text="Ảnh Gốc và Ảnh Chỉnh Sửa", 
                              bg='#ffffff', fg=self.blue_color, font=('Segoe UI', 12, 'bold'))
        title_label.pack(expand=True)
        
        # Sub-frame cho ảnh gốc và hiện tại
        images_subframe = tk.Frame(left_frame, bg='#ffffff')
        images_subframe.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Ảnh gốc (trên)
        orig_title = tk.Label(images_subframe, text="Ảnh Gốc", bg='#ffffff', fg=self.blue_dark, font=('Segoe UI', 10, 'bold'))
        orig_title.pack(pady=(0, 5))
        self.orig_frame = tk.Frame(images_subframe, bg=self.blue_light, highlightthickness=2, highlightbackground=self.blue_color, relief='groove')
        self.orig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)
        self.original_label = tk.Label(self.orig_frame, text="Chưa chọn ảnh", relief=tk.FLAT, bg='#f8f9fa', fg='#6c757d', font=('Segoe UI', 10))
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Ảnh hiện tại (dưới)
        curr_title = tk.Label(images_subframe, text="Ảnh Chỉnh Sửa", bg='#ffffff', fg=self.blue_dark, font=('Segoe UI', 10, 'bold'))
        curr_title.pack(pady=(0, 5))
        self.curr_frame = tk.Frame(images_subframe, bg=self.blue_light, highlightthickness=2, highlightbackground=self.blue_color, relief='groove')
        self.curr_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10)
        self.current_label = tk.Label(self.curr_frame, text="Chưa chỉnh sửa", relief=tk.FLAT, bg='#f8f9fa', fg='#6c757d', font=('Segoe UI', 10))
        self.current_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Thiết lập frame bên phải (công cụ xử lý)
    def _setup_right_frame(self, parent):
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Tiêu đề bên phải
        right_title = tk.Label(right_frame, text="Công Cụ Xử Lý Ảnh", 
                              bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 12, 'bold'))
        right_title.pack(pady=(0, 10))
        
        # Frame chứa các nút chức năng
        self._setup_buttons(right_frame)
        
        # Notebook cho tabs: Biến Đổi, Lọc Ảnh, Khác
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Biến Đổi
        transform_tab = ttk.Frame(self.notebook)
        self.notebook.add(transform_tab, text="🔄 Biến Đổi")
        self.setup_transform_tab(transform_tab)
        
        # Tab 2: Lọc Ảnh
        filter_tab = ttk.Frame(self.notebook)
        self.notebook.add(filter_tab, text="🖼️ Lọc Ảnh")
        self.setup_filter_tab(filter_tab)
        
        # Tab 3: Khác
        other_tab = ttk.Frame(self.notebook)
        self.notebook.add(other_tab, text="⚙️ Khác")
        self.setup_other_tab(other_tab)
    
    # Thiết lập các nút chức năng
    def _setup_buttons(self, parent):
        buttons_frame = tk.Frame(parent, bg='#f0f2f5')
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Nút chọn ảnh
        btn_choose = tk.Button(buttons_frame, text="📁 Chọn Ảnh", command=self.load_image, 
                               bg=self.blue_color, fg='white', font=('Segoe UI', 10, 'bold'),
                               relief='flat', bd=0, padx=15, pady=10,
                               activebackground=self.blue_dark, activeforeground='white',
                               cursor='hand2', highlightthickness=0)
        btn_choose.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        
        # Nút Apply
        btn_apply = tk.Button(buttons_frame, text="✅ Apply", command=self.apply_to_original,
                             bg='#28a745', fg='white', font=('Segoe UI', 10, 'bold'),
                             relief='flat', bd=0, padx=15, pady=10,
                             activebackground='#218838', activeforeground='white',
                             cursor='hand2', highlightthickness=0)
        btn_apply.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        
        # Nút Reset
        btn_reset = tk.Button(buttons_frame, text="🔄 Reset", command=self.reset_image,
                             bg='#ffc107', fg='#212529', font=('Segoe UI', 10, 'bold'),
                             relief='flat', bd=0, padx=15, pady=10,
                             activebackground='#e0a800', activeforeground='#212529',
                             cursor='hand2', highlightthickness=0)
        btn_reset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        
        # Nút Lưu File
        btn_save = tk.Button(buttons_frame, text="💾 Lưu", command=self.save_image,
                            bg='#17a2b8', fg='white', font=('Segoe UI', 10, 'bold'),
                            relief='flat', bd=0, padx=15, pady=10,
                            activebackground='#138496', activeforeground='white',
                            cursor='hand2', highlightthickness=0)
        btn_save.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        
        # Nút Close
        btn_close = tk.Button(buttons_frame, text="❌ Đóng", command=self.root.quit,
                             bg='#dc3545', fg='white', font=('Segoe UI', 10, 'bold'),
                             relief='flat', bd=0, padx=15, pady=10,
                             activebackground='#c82333', activeforeground='white',
                             cursor='hand2', highlightthickness=0)
        btn_close.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    # Thiết lập tab Biến Đổi
    def setup_transform_tab(self, parent):
        # Negative checkbox
        negative_frame = tk.Frame(parent, bg='#f0f2f5', highlightbackground=self.blue_color, 
                                  highlightthickness=1, relief='solid')
        negative_frame.pack(pady=10, fill=tk.X, padx=10)
        self.negative_var = tk.BooleanVar()
        negative_check = tk.Checkbutton(negative_frame, text="Negative Image", variable=self.negative_var, 
                                        command=self.apply_negative, bg='#f0f2f5', fg='#212529', 
                                        font=('Segoe UI', 10), selectcolor=self.gray_light,
                                        activebackground='#f0f2f5', activeforeground=self.blue_color,
                                        cursor='hand2')
        negative_check.pack(padx=15, pady=10)
        
        # Log
        log_frame = self.create_section_frame(parent, "Biến Đổi Log", self.blue_color)
        log_content = tk.Frame(log_frame, bg='#f0f2f5')
        log_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(log_content, text="Hệ số C (0.1-2.0):", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.log_c_var = tk.DoubleVar(value=1.0)
        log_scale = ttk.Scale(log_content, from_=0.1, to=2.0, variable=self.log_c_var, orient=tk.HORIZONTAL, command=self.on_log_change, length=200)
        log_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        log_value_label = tk.Label(log_content, text="1.0", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        log_value_label.pack(side=tk.LEFT)
        self.log_value_label = log_value_label
        
        # Piecewise-Linear
        piecewise_frame = self.create_section_frame(parent, "Biến Đổi Piecewise-Linear", self.blue_light)
        piecewise_content1 = tk.Frame(piecewise_frame, bg='#f0f2f5')
        piecewise_content1.pack(fill=tk.X, padx=15, pady=(0, 5))
        tk.Label(piecewise_content1, text="Hệ số Cao:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.piecewise_high_var = tk.DoubleVar(value=1.0)
        piecewise_high_scale = ttk.Scale(piecewise_content1, from_=0.1, to=3.0, variable=self.piecewise_high_var, orient=tk.HORIZONTAL, command=self.on_piecewise_change, length=200)
        piecewise_high_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        piecewise_high_label = tk.Label(piecewise_content1, text="1.0", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        piecewise_high_label.pack(side=tk.LEFT)
        self.piecewise_high_label = piecewise_high_label
        
        piecewise_content2 = tk.Frame(piecewise_frame, bg='#f0f2f5')
        piecewise_content2.pack(fill=tk.X, padx=15, pady=5)
        tk.Label(piecewise_content2, text="Hệ số Thấp:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.piecewise_low_var = tk.DoubleVar(value=0.5)
        piecewise_low_scale = ttk.Scale(piecewise_content2, from_=0.1, to=1.0, variable=self.piecewise_low_var, orient=tk.HORIZONTAL, command=self.on_piecewise_change, length=200)
        piecewise_low_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        piecewise_low_label = tk.Label(piecewise_content2, text="0.5", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        piecewise_low_label.pack(side=tk.LEFT)
        self.piecewise_low_label = piecewise_low_label
        
        # Gamma
        gamma_frame = self.create_section_frame(parent, "Biến Đổi Gamma", self.blue_dark)
        gamma_content1 = tk.Frame(gamma_frame, bg='#f0f2f5')
        gamma_content1.pack(fill=tk.X, padx=15, pady=(0, 5))
        tk.Label(gamma_content1, text="Hệ số C (0.1-2.0):", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.gamma_c_var = tk.DoubleVar(value=1.0)
        gamma_c_scale = ttk.Scale(gamma_content1, from_=0.1, to=2.0, variable=self.gamma_c_var, orient=tk.HORIZONTAL, command=self.on_gamma_change, length=200)
        gamma_c_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        gamma_c_label = tk.Label(gamma_content1, text="1.0", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        gamma_c_label.pack(side=tk.LEFT)
        self.gamma_c_label = gamma_c_label
        
        gamma_content2 = tk.Frame(gamma_frame, bg='#f0f2f5')
        gamma_content2.pack(fill=tk.X, padx=15, pady=5)
        tk.Label(gamma_content2, text="Gamma:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.gamma_var = tk.DoubleVar(value=1.0)
        gamma_scale = ttk.Scale(gamma_content2, from_=0.1, to=3.0, variable=self.gamma_var, orient=tk.HORIZONTAL, command=self.on_gamma_change, length=200)
        gamma_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        gamma_label = tk.Label(gamma_content2, text="1.0", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        gamma_label.pack(side=tk.LEFT)
        self.gamma_label = gamma_label
    
    # Thiết lập tab Lọc Ảnh
    def setup_filter_tab(self, parent):
        # Frame chọn miền
        domain_frame = tk.Frame(parent, bg='#f0f2f5', highlightbackground=self.blue_color, 
                               highlightthickness=2, relief='groove')
        domain_frame.pack(fill=tk.X, padx=10, pady=10)
        
        domain_inner = tk.Frame(domain_frame, bg='#ffffff')
        domain_inner.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(domain_inner, text="Chọn miền xử lý:", bg='#ffffff', fg='#212529', 
                font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=15, pady=10)
        
        tk.Radiobutton(domain_inner, text="Miền Không Gian", variable=self.domain_var, 
                      value=0, command=self.on_domain_change, bg='#ffffff', fg='#212529',
                      font=('Segoe UI', 10), selectcolor=self.gray_light,
                      activebackground='#ffffff', activeforeground=self.blue_color,
                      cursor='hand2').pack(side=tk.LEFT, padx=10)
        
        tk.Radiobutton(domain_inner, text="Miền Tần Số", variable=self.domain_var, 
                      value=1, command=self.on_domain_change, bg='#ffffff', fg='#212529',
                      font=('Segoe UI', 10), selectcolor=self.gray_light,
                      activebackground='#ffffff', activeforeground=self.blue_color,
                      cursor='hand2').pack(side=tk.LEFT, padx=10)
        
        # Nút so sánh Gaussian
        compare_btn = tk.Button(domain_inner, text="⚖️ So Sánh Gaussian", 
                               command=self.compare_gaussian_filters,
                               bg='#17a2b8', fg='#ffffff', font=('Segoe UI', 10, 'bold'),
                               relief='raised', bd=2, cursor='hand2', padx=15, pady=5)
        compare_btn.pack(side=tk.LEFT, padx=(20, 15))
        
        # Container chứa canvas và timing info
        content_container = tk.Frame(parent, bg='#f0f2f5')
        content_container.pack(fill=tk.BOTH, expand=True)
        
        # Tạo Canvas với Scrollbar để có thể cuộn
        canvas_frame = tk.Frame(content_container, bg='#f0f2f5')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas để chứa nội dung có thể cuộn
        canvas = tk.Canvas(canvas_frame, bg='#f0f2f5', highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        
        # Frame bên trong canvas để chứa tất cả các filter
        scrollable_frame = tk.Frame(canvas, bg='#f0f2f5')
        
        # Tạo window trong canvas cho scrollable_frame
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Cấu hình scrollbar
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Cấu hình canvas để resize với window
        def configure_canvas_width(e):
            canvas_width = e.width
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        canvas.bind('<Configure>', configure_canvas_width)
        
        # Bind mousewheel để cuộn
        def on_mousewheel(event):
            if canvas.winfo_containing(event.x_root, event.y_root) == canvas:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)
        
        # Đặt canvas và scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Lưu reference đến scrollable_frame
        self.filter_scrollable_frame = scrollable_frame
        
        # Frame cho lọc không gian
        self.spatial_filter_frame = tk.Frame(scrollable_frame, bg='#f0f2f5')
        
        # Frame cho lọc tần số
        self.frequency_filter_frame = tk.Frame(scrollable_frame, bg='#f0f2f5')
        
        # Thiết lập các filter không gian
        self._setup_spatial_filters()
        
        # Thiết lập các filter tần số
        self._setup_frequency_filters()
        
        # Frame hiển thị thông tin thời gian cho miền tần số
        self.timing_info_frame = self._setup_timing_info_frame(content_container)
        
        # Frame hiển thị thông tin thời gian cho miền không gian (chỉ tổng thời gian)
        self.spatial_timing_info_frame = self._setup_spatial_timing_info_frame(content_container)
        
        # Hiển thị frame phù hợp
        self.on_domain_change()
    
    # Thiết lập các bộ lọc không gian
    def _setup_spatial_filters(self):
        # Average filter
        self._setup_average_filter(self.spatial_filter_frame)
        
        # Gaussian filter
        self._setup_gaussian_filter(self.spatial_filter_frame)
        
        # Median filter
        self._setup_median_filter(self.spatial_filter_frame)
        
        # Max filter
        self._setup_max_filter(self.spatial_filter_frame)
        
        # Min filter
        self._setup_min_filter(self.spatial_filter_frame)
        
        # Midpoint filter
        self._setup_midpoint_filter(self.spatial_filter_frame)
    
    # Thiết lập các bộ lọc tần số
    def _setup_frequency_filters(self):
        # Lọc thông thấp lý tưởng
        self._setup_ideal_lowpass_filter(self.frequency_filter_frame)
        
        # Lọc thông thấp Gaussian
        self._setup_gaussian_lowpass_filter(self.frequency_filter_frame)
        
        # Lọc thông thấp Butterworth
        self._setup_butterworth_lowpass_filter(self.frequency_filter_frame)
        
        # Lọc thông cao lý tưởng
        self._setup_ideal_highpass_filter(self.frequency_filter_frame)
        
        # Lọc thông cao Gaussian
        self._setup_gaussian_highpass_filter(self.frequency_filter_frame)
        
        # Lọc thông cao Butterworth
        self._setup_butterworth_highpass_filter(self.frequency_filter_frame)
    
    # Xử lý khi thay đổi miền
    def on_domain_change(self):
        if self.domain_var.get() == 0:  # Miền không gian
            self.frequency_filter_frame.pack_forget()
            self.timing_info_frame.pack_forget()
            self.spatial_filter_frame.pack(fill=tk.BOTH, expand=True)
            # Hiển thị timing info frame cho miền không gian
            self.spatial_timing_info_frame.pack(fill=tk.X, padx=10, pady=(5, 10), side=tk.BOTTOM)
        else:  # Miền tần số
            self.spatial_filter_frame.pack_forget()
            self.spatial_timing_info_frame.pack_forget()
            self.frequency_filter_frame.pack(fill=tk.BOTH, expand=True)
            # Hiển thị timing info frame ở dưới cùng, không expand để đảm bảo hiển thị đầy đủ
            self.timing_info_frame.pack(fill=tk.X, padx=10, pady=(5, 10), side=tk.BOTTOM)
    
    # Thiết lập frame hiển thị thông tin thời gian
    def _setup_timing_info_frame(self, parent):
        timing_frame = tk.Frame(parent, bg='#f0f2f5', highlightbackground='#17a2b8', 
                               highlightthickness=2, relief='groove')
        
        timing_inner = tk.Frame(timing_frame, bg='#ffffff')
        timing_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        title_label = tk.Label(timing_inner, text="⏱️ Thông Tin Thời Gian Xử Lý", 
                              bg='#ffffff', fg='#17a2b8', font=('Segoe UI', 11, 'bold'))
        title_label.pack(pady=(10, 5))
        
        separator = tk.Frame(timing_inner, bg='#17a2b8', height=2)
        separator.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        # Tạo Canvas với Scrollbar cho phần thông tin chi tiết
        timing_canvas_frame = tk.Frame(timing_inner, bg='#ffffff')
        timing_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))
        
        timing_canvas = tk.Canvas(timing_canvas_frame, bg='#ffffff', highlightthickness=0)
        timing_scrollbar = ttk.Scrollbar(timing_canvas_frame, orient="vertical", command=timing_canvas.yview)
        
        # Frame bên trong canvas để chứa thông tin chi tiết
        info_container = tk.Frame(timing_canvas, bg='#ffffff')
        
        # Tạo window trong canvas cho info_container
        timing_canvas_window = timing_canvas.create_window((0, 0), window=info_container, anchor="nw")
        
        # Cấu hình scrollbar
        def configure_timing_scrollregion(e):
            timing_canvas.configure(scrollregion=timing_canvas.bbox("all"))
        
        info_container.bind("<Configure>", configure_timing_scrollregion)
        
        # Cấu hình canvas để resize với window
        def configure_timing_canvas_width(e):
            canvas_width = e.width
            timing_canvas.itemconfig(timing_canvas_window, width=canvas_width)
        
        timing_canvas.bind('<Configure>', configure_timing_canvas_width)
        
        # Bind mousewheel để cuộn
        def on_timing_mousewheel(event):
            if timing_canvas.winfo_containing(event.x_root, event.y_root) == timing_canvas:
                timing_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def bind_timing_mousewheel(event):
            timing_canvas.bind_all("<MouseWheel>", on_timing_mousewheel)
        
        def unbind_timing_mousewheel(event):
            timing_canvas.unbind_all("<MouseWheel>")
        
        timing_canvas.bind("<Enter>", bind_timing_mousewheel)
        timing_canvas.bind("<Leave>", unbind_timing_mousewheel)
        
        # Đặt canvas và scrollbar
        timing_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        timing_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        timing_canvas.configure(yscrollcommand=timing_scrollbar.set)
        
        # Tạo các label để hiển thị thông tin
        self.timing_filter_name_label = tk.Label(info_container, text="Chưa có dữ liệu", 
                                                 bg='#ffffff', fg='#212529', 
                                                 font=('Segoe UI', 10, 'bold'), anchor='w')
        self.timing_filter_name_label.pack(fill=tk.X, pady=(0, 5))
        
        self.timing_params_label = tk.Label(info_container, text="", 
                                           bg='#ffffff', fg='#6c757d', 
                                           font=('Segoe UI', 9), anchor='w')
        self.timing_params_label.pack(fill=tk.X, pady=(0, 10))
        
        # Frame chứa 6 bước
        steps_frame = tk.Frame(info_container, bg='#f8f9fa', relief='solid', bd=1)
        steps_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.timing_b1_label = tk.Label(steps_frame, text="B1 - Tiền xử lý:        -- ms", 
                                       bg='#f8f9fa', fg='#212529', font=('Segoe UI', 9), anchor='w')
        self.timing_b1_label.pack(fill=tk.X, padx=10, pady=2)
        
        self.timing_b2_label = tk.Label(steps_frame, text="B2 - FFT:               -- ms", 
                                       bg='#f8f9fa', fg='#212529', font=('Segoe UI', 9), anchor='w')
        self.timing_b2_label.pack(fill=tk.X, padx=10, pady=2)
        
        self.timing_b3_label = tk.Label(steps_frame, text="B3 - Xây dựng bộ lọc:   -- ms", 
                                       bg='#f8f9fa', fg='#212529', font=('Segoe UI', 9), anchor='w')
        self.timing_b3_label.pack(fill=tk.X, padx=10, pady=2)
        
        self.timing_b4_label = tk.Label(steps_frame, text="B4 - Lọc (F × H):       -- ms", 
                                       bg='#f8f9fa', fg='#212529', font=('Segoe UI', 9), anchor='w')
        self.timing_b4_label.pack(fill=tk.X, padx=10, pady=2)
        
        self.timing_b5_label = tk.Label(steps_frame, text="B5 - IFFT:              -- ms", 
                                       bg='#f8f9fa', fg='#212529', font=('Segoe UI', 9), anchor='w')
        self.timing_b5_label.pack(fill=tk.X, padx=10, pady=2)
        
        self.timing_b6_label = tk.Label(steps_frame, text="B6 - Hậu xử lý:         -- ms", 
                                       bg='#f8f9fa', fg='#212529', font=('Segoe UI', 9), anchor='w')
        self.timing_b6_label.pack(fill=tk.X, padx=10, pady=2)
        
        # Separator trước tổng thời gian
        total_separator = tk.Frame(info_container, bg='#dee2e6', height=1)
        total_separator.pack(fill=tk.X, pady=(10, 5))
        
        # Frame riêng cho tổng thời gian để làm nổi bật
        total_frame = tk.Frame(info_container, bg='#e7f3ff', relief='solid', bd=1)
        total_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Tổng thời gian
        self.timing_total_label = tk.Label(total_frame, text="Tổng thời gian:         -- ms", 
                                           bg='#e7f3ff', fg='#0056b3', 
                                           font=('Segoe UI', 11, 'bold'), anchor='w')
        self.timing_total_label.pack(fill=tk.X, padx=15, pady=8)
        
        return timing_frame
    
    # Thiết lập frame hiển thị thông tin thời gian cho miền không gian (chỉ tổng thời gian)
    def _setup_spatial_timing_info_frame(self, parent):
        timing_frame = tk.Frame(parent, bg='#f0f2f5', highlightbackground='#28a745', 
                               highlightthickness=2, relief='groove')
        
        timing_inner = tk.Frame(timing_frame, bg='#ffffff')
        timing_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        title_label = tk.Label(timing_inner, text="⏱️ Thông Tin Thời Gian Xử Lý", 
                              bg='#ffffff', fg='#28a745', font=('Segoe UI', 11, 'bold'))
        title_label.pack(pady=(10, 5))
        
        separator = tk.Frame(timing_inner, bg='#28a745', height=2)
        separator.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        # Frame chứa thông tin
        info_container = tk.Frame(timing_inner, bg='#ffffff')
        info_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Label hiển thị tên bộ lọc
        self.spatial_timing_filter_name_label = tk.Label(info_container, text="Chưa có dữ liệu", 
                                                         bg='#ffffff', fg='#212529', 
                                                         font=('Segoe UI', 10, 'bold'), anchor='w')
        self.spatial_timing_filter_name_label.pack(fill=tk.X, pady=(0, 5))
        
        # Label hiển thị tham số
        self.spatial_timing_params_label = tk.Label(info_container, text="", 
                                                     bg='#ffffff', fg='#6c757d', 
                                                     font=('Segoe UI', 9), anchor='w')
        self.spatial_timing_params_label.pack(fill=tk.X, pady=(0, 10))
        
        # Frame riêng cho tổng thời gian để làm nổi bật
        total_frame = tk.Frame(info_container, bg='#d4edda', relief='solid', bd=1)
        total_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Tổng thời gian
        self.spatial_timing_total_label = tk.Label(total_frame, text="Tổng thời gian:         -- ms", 
                                                   bg='#d4edda', fg='#155724', 
                                                   font=('Segoe UI', 11, 'bold'), anchor='w')
        self.spatial_timing_total_label.pack(fill=tk.X, padx=15, pady=8)
        
        return timing_frame
    
    # Cập nhật hiển thị thông tin thời gian cho miền tần số
    def _update_timing_info(self, timing_info):
        if timing_info:
            self.timing_filter_name_label.config(text=timing_info['filter_name'])
            self.timing_params_label.config(text=f"Tham số: {timing_info['params']}")
            self.timing_b1_label.config(text=f"B1 - Tiền xử lý:        {timing_info['time_b1']:.3f} ms")
            self.timing_b2_label.config(text=f"B2 - FFT:               {timing_info['time_b2']:.3f} ms")
            self.timing_b3_label.config(text=f"B3 - Xây dựng bộ lọc:   {timing_info['time_b3']:.3f} ms")
            self.timing_b4_label.config(text=f"B4 - Lọc (F × H):       {timing_info['time_b4']:.3f} ms")
            self.timing_b5_label.config(text=f"B5 - IFFT:              {timing_info['time_b5']:.3f} ms")
            self.timing_b6_label.config(text=f"B6 - Hậu xử lý:         {timing_info['time_b6']:.3f} ms")
            # Cập nhật tổng thời gian với định dạng nổi bật
            total_text = f"Tổng thời gian:         {timing_info['total_time']:.3f} ms"
            self.timing_total_label.config(text=total_text)
            # Đảm bảo label được hiển thị
            self.timing_total_label.update()
    
    # Cập nhật hiển thị thông tin thời gian cho miền không gian (chỉ tổng thời gian)
    def _update_spatial_timing_info(self, filter_name, params, total_time):
        self.spatial_timing_filter_name_label.config(text=filter_name)
        self.spatial_timing_params_label.config(text=f"Tham số: {params}")
        total_text = f"Tổng thời gian:         {total_time:.3f} ms"
        self.spatial_timing_total_label.config(text=total_text)
        self.spatial_timing_total_label.update()
    
    # Thiết lập bộ lọc trung bình
    def _setup_average_filter(self, parent):
        avg_frame = self.create_section_frame(parent, "Lọc Trung Bình", '#28a745')
        avg_content = tk.Frame(avg_frame, bg='#f0f2f5')
        avg_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(avg_content, text="Kích thước:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.avg_size_var = tk.IntVar(value=3)
        avg_size_scale = ttk.Scale(avg_content, from_=3, to=15, variable=self.avg_size_var, orient=tk.HORIZONTAL, command=self.on_avg_change, length=200)
        avg_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        avg_size_label = tk.Label(avg_content, text="3", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        avg_size_label.pack(side=tk.LEFT)
        self.avg_size_label = avg_size_label
    
    # Thiết lập bộ lọc Gaussian
    def _setup_gaussian_filter(self, parent):
        gauss_frame = self.create_section_frame(parent, "Lọc Gaussian", '#ffc107')
        gauss_content1 = tk.Frame(gauss_frame, bg='#f0f2f5')
        gauss_content1.pack(fill=tk.X, padx=15, pady=(0, 5))
        tk.Label(gauss_content1, text="Kích thước:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.gauss_size_var = tk.IntVar(value=3)
        gauss_size_scale = ttk.Scale(gauss_content1, from_=3, to=15, variable=self.gauss_size_var, orient=tk.HORIZONTAL, command=self.on_gauss_change, length=200)
        gauss_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        gauss_size_label = tk.Label(gauss_content1, text="3", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        gauss_size_label.pack(side=tk.LEFT)
        self.gauss_size_label = gauss_size_label
        
        gauss_content2 = tk.Frame(gauss_frame, bg='#f0f2f5')
        gauss_content2.pack(fill=tk.X, padx=15, pady=5)
        tk.Label(gauss_content2, text="Sigma:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.gauss_sigma_var = tk.DoubleVar(value=1.0)
        gauss_sigma_scale = ttk.Scale(gauss_content2, from_=0.1, to=5.0, variable=self.gauss_sigma_var, orient=tk.HORIZONTAL, command=self.on_gauss_change, length=200)
        gauss_sigma_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        gauss_sigma_label = tk.Label(gauss_content2, text="1.0", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        gauss_sigma_label.pack(side=tk.LEFT)
        self.gauss_sigma_label = gauss_sigma_label
    
    # Thiết lập bộ lọc trung vị
    def _setup_median_filter(self, parent):
        median_frame = self.create_section_frame(parent, "Lọc Trung Vị", '#fd7e14')
        median_content = tk.Frame(median_frame, bg='#f0f2f5')
        median_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(median_content, text="Kích thước:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.median_size_var = tk.IntVar(value=3)
        median_size_scale = ttk.Scale(median_content, from_=3, to=15, variable=self.median_size_var, orient=tk.HORIZONTAL, command=self.on_median_change, length=200)
        median_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        median_size_label = tk.Label(median_content, text="3", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        median_size_label.pack(side=tk.LEFT)
        self.median_size_label = median_size_label
    
    # Thiết lập bộ lọc Max
    def _setup_max_filter(self, parent):
        max_frame = self.create_section_frame(parent, "Lọc Max", '#dc3545')
        max_content = tk.Frame(max_frame, bg='#f0f2f5')
        max_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(max_content, text="Kích thước:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.max_size_var = tk.IntVar(value=3)
        max_size_scale = ttk.Scale(max_content, from_=3, to=15, variable=self.max_size_var, orient=tk.HORIZONTAL, command=self.on_max_change, length=200)
        max_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        max_size_label = tk.Label(max_content, text="3", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        max_size_label.pack(side=tk.LEFT)
        self.max_size_label = max_size_label
    
    # Thiết lập bộ lọc Min
    def _setup_min_filter(self, parent):
        min_frame = self.create_section_frame(parent, "Lọc Min", '#6f42c1')
        min_content = tk.Frame(min_frame, bg='#f0f2f5')
        min_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(min_content, text="Kích thước:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.min_size_var = tk.IntVar(value=3)
        min_size_scale = ttk.Scale(min_content, from_=3, to=15, variable=self.min_size_var, orient=tk.HORIZONTAL, command=self.on_min_change, length=200)
        min_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        min_size_label = tk.Label(min_content, text="3", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        min_size_label.pack(side=tk.LEFT)
        self.min_size_label = min_size_label
    
    # Thiết lập bộ lọc Midpoint
    def _setup_midpoint_filter(self, parent):
        midpoint_frame = self.create_section_frame(parent, "Lọc Midpoint", '#20c997')
        midpoint_content = tk.Frame(midpoint_frame, bg='#f0f2f5')
        midpoint_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(midpoint_content, text="Kích thước:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.midpoint_size_var = tk.IntVar(value=3)
        midpoint_size_scale = ttk.Scale(midpoint_content, from_=3, to=15, variable=self.midpoint_size_var, orient=tk.HORIZONTAL, command=self.on_midpoint_change, length=200)
        midpoint_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        midpoint_size_label = tk.Label(midpoint_content, text="3", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        midpoint_size_label.pack(side=tk.LEFT)
        self.midpoint_size_label = midpoint_size_label
    
    # Thiết lập các bộ lọc tần số
    def _setup_ideal_lowpass_filter(self, parent):
        ideal_lp_frame = self.create_section_frame(parent, "Lọc Thông Thấp Lý Tưởng", '#28a745')
        ideal_lp_content = tk.Frame(ideal_lp_frame, bg='#f0f2f5')
        ideal_lp_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(ideal_lp_content, text="D0:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.ideal_lp_d0_var = tk.DoubleVar(value=50)
        ideal_lp_d0_scale = ttk.Scale(ideal_lp_content, from_=1, to=200, variable=self.ideal_lp_d0_var, orient=tk.HORIZONTAL, command=self.on_ideal_lp_change, length=200)
        ideal_lp_d0_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        ideal_lp_d0_label = tk.Label(ideal_lp_content, text="50", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        ideal_lp_d0_label.pack(side=tk.LEFT)
        self.ideal_lp_d0_label = ideal_lp_d0_label
    
    def _setup_gaussian_lowpass_filter(self, parent):
        gauss_lp_frame = self.create_section_frame(parent, "Lọc Thông Thấp Gaussian", '#ffc107')
        gauss_lp_content = tk.Frame(gauss_lp_frame, bg='#f0f2f5')
        gauss_lp_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(gauss_lp_content, text="D0:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.gauss_lp_d0_var = tk.DoubleVar(value=50)
        gauss_lp_d0_scale = ttk.Scale(gauss_lp_content, from_=1, to=200, variable=self.gauss_lp_d0_var, orient=tk.HORIZONTAL, command=self.on_gauss_lp_change, length=200)
        gauss_lp_d0_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        gauss_lp_d0_label = tk.Label(gauss_lp_content, text="50", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        gauss_lp_d0_label.pack(side=tk.LEFT)
        self.gauss_lp_d0_label = gauss_lp_d0_label
    
    def _setup_butterworth_lowpass_filter(self, parent):
        butter_lp_frame = self.create_section_frame(parent, "Lọc Thông Thấp Butterworth", '#fd7e14')
        butter_lp_content1 = tk.Frame(butter_lp_frame, bg='#f0f2f5')
        butter_lp_content1.pack(fill=tk.X, padx=15, pady=(0, 5))
        tk.Label(butter_lp_content1, text="D0:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.butter_lp_d0_var = tk.DoubleVar(value=75)
        butter_lp_d0_scale = ttk.Scale(butter_lp_content1, from_=1, to=200, variable=self.butter_lp_d0_var, orient=tk.HORIZONTAL, command=self.on_butter_lp_change, length=200)
        butter_lp_d0_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        butter_lp_d0_label = tk.Label(butter_lp_content1, text="75", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        butter_lp_d0_label.pack(side=tk.LEFT)
        self.butter_lp_d0_label = butter_lp_d0_label
        
        butter_lp_content2 = tk.Frame(butter_lp_frame, bg='#f0f2f5')
        butter_lp_content2.pack(fill=tk.X, padx=15, pady=5)
        tk.Label(butter_lp_content2, text="n:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.butter_lp_n_var = tk.DoubleVar(value=2)
        butter_lp_n_scale = ttk.Scale(butter_lp_content2, from_=1, to=10, variable=self.butter_lp_n_var, orient=tk.HORIZONTAL, command=self.on_butter_lp_change, length=200)
        butter_lp_n_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        butter_lp_n_label = tk.Label(butter_lp_content2, text="2", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        butter_lp_n_label.pack(side=tk.LEFT)
        self.butter_lp_n_label = butter_lp_n_label
    
    def _setup_ideal_highpass_filter(self, parent):
        ideal_hp_frame = self.create_section_frame(parent, "Lọc Thông Cao Lý Tưởng", '#dc3545')
        ideal_hp_content = tk.Frame(ideal_hp_frame, bg='#f0f2f5')
        ideal_hp_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(ideal_hp_content, text="D0:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.ideal_hp_d0_var = tk.DoubleVar(value=10)
        ideal_hp_d0_scale = ttk.Scale(ideal_hp_content, from_=1, to=100, variable=self.ideal_hp_d0_var, orient=tk.HORIZONTAL, command=self.on_ideal_hp_change, length=200)
        ideal_hp_d0_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        ideal_hp_d0_label = tk.Label(ideal_hp_content, text="10", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        ideal_hp_d0_label.pack(side=tk.LEFT)
        self.ideal_hp_d0_label = ideal_hp_d0_label
    
    def _setup_gaussian_highpass_filter(self, parent):
        gauss_hp_frame = self.create_section_frame(parent, "Lọc Thông Cao Gaussian", '#6f42c1')
        gauss_hp_content = tk.Frame(gauss_hp_frame, bg='#f0f2f5')
        gauss_hp_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(gauss_hp_content, text="D0:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.gauss_hp_d0_var = tk.DoubleVar(value=25)
        gauss_hp_d0_scale = ttk.Scale(gauss_hp_content, from_=1, to=100, variable=self.gauss_hp_d0_var, orient=tk.HORIZONTAL, command=self.on_gauss_hp_change, length=200)
        gauss_hp_d0_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        gauss_hp_d0_label = tk.Label(gauss_hp_content, text="25", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        gauss_hp_d0_label.pack(side=tk.LEFT)
        self.gauss_hp_d0_label = gauss_hp_d0_label
    
    def _setup_butterworth_highpass_filter(self, parent):
        butter_hp_frame = self.create_section_frame(parent, "Lọc Thông Cao Butterworth", '#20c997')
        butter_hp_content1 = tk.Frame(butter_hp_frame, bg='#f0f2f5')
        butter_hp_content1.pack(fill=tk.X, padx=15, pady=(0, 5))
        tk.Label(butter_hp_content1, text="D0:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.butter_hp_d0_var = tk.DoubleVar(value=20)
        butter_hp_d0_scale = ttk.Scale(butter_hp_content1, from_=1, to=100, variable=self.butter_hp_d0_var, orient=tk.HORIZONTAL, command=self.on_butter_hp_change, length=200)
        butter_hp_d0_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        butter_hp_d0_label = tk.Label(butter_hp_content1, text="20", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        butter_hp_d0_label.pack(side=tk.LEFT)
        self.butter_hp_d0_label = butter_hp_d0_label
        
        butter_hp_content2 = tk.Frame(butter_hp_frame, bg='#f0f2f5')
        butter_hp_content2.pack(fill=tk.X, padx=15, pady=5)
        tk.Label(butter_hp_content2, text="n:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.butter_hp_n_var = tk.DoubleVar(value=2)
        butter_hp_n_scale = ttk.Scale(butter_hp_content2, from_=1, to=10, variable=self.butter_hp_n_var, orient=tk.HORIZONTAL, command=self.on_butter_hp_change, length=200)
        butter_hp_n_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        butter_hp_n_label = tk.Label(butter_hp_content2, text="2", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        butter_hp_n_label.pack(side=tk.LEFT)
        self.butter_hp_n_label = butter_hp_n_label
    
    # Thiết lập tab Khác
    def setup_other_tab(self, parent):
        # Histogram
        hist_frame = self.create_section_frame(parent, "Cân Bằng Histogram", '#6f42c1')
        hist_content = tk.Frame(hist_frame, bg='#f0f2f5')
        hist_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(hist_content, text="Độ sáng:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.hist_value_var = tk.DoubleVar(value=1.0)
        hist_value_scale = ttk.Scale(hist_content, from_=0.1, to=2.0, variable=self.hist_value_var, orient=tk.HORIZONTAL, command=self.on_hist_change, length=200)
        hist_value_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        hist_value_label = tk.Label(hist_content, text="1.0", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        hist_value_label.pack(side=tk.LEFT)
        self.hist_value_label = hist_value_label
    
    # Tạo frame section với viền màu nổi bật
    def create_section_frame(self, parent, title, color):
        outer_frame = tk.Frame(parent, bg='#f0f2f5', highlightbackground=color, 
                               highlightthickness=2, relief='groove')
        outer_frame.pack(fill=tk.X, pady=10, padx=10)
        
        inner_frame = tk.Frame(outer_frame, bg='#ffffff', relief='raised', bd=1)
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        title_label = tk.Label(inner_frame, text=title, bg='#ffffff', fg=color, 
                               font=('Segoe UI', 10, 'bold'), anchor='w')
        title_label.pack(fill=tk.X, padx=15, pady=(10, 5))
        
        separator = tk.Frame(inner_frame, bg=color, height=2)
        separator.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        return inner_frame
    
    # Tải ảnh từ file
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        if file_path:
            img_bgr = cv2.imread(file_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                self.original_image = Image.fromarray(img_rgb)
                self.current_image = self.original_image.copy()
                self.active_transformation = None
            else:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh từ file này.")
                return
            # Reset các giá trị về mặc định
            self._reset_all_values()
            self.display_images()
    
    # Reset tất cả các giá trị về mặc định
    def _reset_all_values(self):
        self.negative_var.set(False)
        self.log_c_var.set(1.0)
        self.piecewise_high_var.set(1.0)
        self.piecewise_low_var.set(0.5)
        self.gamma_c_var.set(1.0)
        self.gamma_var.set(1.0)
        self.avg_size_var.set(3)
        self.gauss_size_var.set(3)
        self.gauss_sigma_var.set(1.0)
        self.median_size_var.set(3)
        self.max_size_var.set(3)
        self.min_size_var.set(3)
        self.midpoint_size_var.set(3)
        self.hist_value_var.set(1.0)
        # Reset các giá trị lọc tần số
        self.ideal_lp_d0_var.set(50)
        self.gauss_lp_d0_var.set(50)
        self.butter_lp_d0_var.set(75)
        self.butter_lp_n_var.set(2)
        self.ideal_hp_d0_var.set(10)
        self.gauss_hp_d0_var.set(25)
        self.butter_hp_d0_var.set(20)
        self.butter_hp_n_var.set(2)
        # Cập nhật labels
        self.log_value_label.config(text="1.0")
        self.piecewise_high_label.config(text="1.0")
        self.piecewise_low_label.config(text="0.5")
        self.gamma_c_label.config(text="1.0")
        self.gamma_label.config(text="1.0")
        self.avg_size_label.config(text="3")
        self.gauss_size_label.config(text="3")
        self.gauss_sigma_label.config(text="1.0")
        self.median_size_label.config(text="3")
        self.max_size_label.config(text="3")
        self.min_size_label.config(text="3")
        self.midpoint_size_label.config(text="3")
        self.hist_value_label.config(text="1.0")
        # Cập nhật labels lọc tần số
        if hasattr(self, 'ideal_lp_d0_label'):
            self.ideal_lp_d0_label.config(text="50")
            self.gauss_lp_d0_label.config(text="50")
            self.butter_lp_d0_label.config(text="75")
            self.butter_lp_n_label.config(text="2")
            self.ideal_hp_d0_label.config(text="10")
            self.gauss_hp_d0_label.config(text="25")
            self.butter_hp_d0_label.config(text="20")
            self.butter_hp_n_label.config(text="2")
    
    # Resize ảnh giữ nguyên tỷ lệ để fit vào kích thước cho trước
    def resize_image_to_fit(self, image, max_width, max_height):
        if image is None:
            return None
        
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        
        if original_width > max_width or original_height > max_height:
            if max_width / max_height > aspect_ratio:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
        else:
            new_width = original_width
            new_height = original_height
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Hiển thị ảnh
    def display_images(self):
        if self.original_image:
            self.root.update_idletasks()
            max_width, max_height = self._get_display_size()
            
            orig_resized = self.resize_image_to_fit(self.original_image, max_width, max_height)
            if orig_resized:
                self.photo_original = ImageTk.PhotoImage(orig_resized)
                self.original_label.configure(image=self.photo_original, text="")
            
            if self.current_image:
                curr_resized = self.resize_image_to_fit(self.current_image, max_width, max_height)
                if curr_resized:
                    self.photo_current = ImageTk.PhotoImage(curr_resized)
                    self.current_label.configure(image=self.photo_current, text="")
            else:
                self.current_label.configure(image="", text="Chưa chỉnh sửa")
            
            self.root.after(50, self._refresh_images)
    
    # Lấy kích thước hiển thị cho ảnh dựa trên frame thực tế
    def _get_display_size(self):
        try:
            frame_width = self.orig_frame.winfo_width()
            frame_height = self.orig_frame.winfo_height()
            
            if frame_width > 30 and frame_height > 30:
                display_width = frame_width - 40
                display_height = frame_height - 40
                return (max(100, display_width), max(100, display_height))
        except:
            pass
        
        try:
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            
            if window_width > 200 and window_height > 200:
                estimated_width = int(window_width * 0.50) - 60
                estimated_height = int((window_height - 200) * 0.42) - 40
                return (max(300, estimated_width), max(200, estimated_height))
        except:
            pass
        
        return (500, 350)
    
    # Làm mới kích thước ảnh sau khi frame được render hoàn toàn
    def _refresh_images(self):
        if self.original_image:
            max_width, max_height = self._get_display_size()
            
            try:
                orig_resized = self.resize_image_to_fit(self.original_image, max_width, max_height)
                if orig_resized:
                    self.photo_original = ImageTk.PhotoImage(orig_resized)
                    self.original_label.configure(image=self.photo_original, text="")
                
                if self.current_image:
                    curr_resized = self.resize_image_to_fit(self.current_image, max_width, max_height)
                    if curr_resized:
                        self.photo_current = ImageTk.PhotoImage(curr_resized)
                        self.current_label.configure(image=self.photo_current, text="")
            except Exception:
                pass
    
    # Xử lý ảnh ở background thread để không làm đóng băng GUI
    def _process_image_async(self, process_func, *args, **kwargs):
        if self.is_processing:
            return
        
        def process():
            self.is_processing = True
            try:
                # Chạy hàm xử lý ảnh
                result = process_func(*args, **kwargs)
                # Cập nhật GUI ở main thread
                self.root.after(0, lambda: self._update_image(result))
            except Exception as e:
                print(f"Lỗi xử lý ảnh: {e}")
                self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi xử lý ảnh: {str(e)}"))
            finally:
                self.is_processing = False
        
        # Tạo và chạy thread
        self.processing_thread = threading.Thread(target=process, daemon=True)
        self.processing_thread.start()
    
    # Cập nhật ảnh hiện tại (phải được gọi từ main thread)
    def _update_image(self, image):
        if image is not None:
            self.current_image = image
            self.display_images()
    
    # Debounce function để tránh gọi quá nhiều lần
    def _debounce(self, func, delay=None):
        if delay is None:
            delay = int(self.debounce_delay * 1000)  # Convert to ms
        
        # Hủy timer cũ nếu có
        if self.debounce_timer is not None:
            self.root.after_cancel(self.debounce_timer)
        
        # Tạo timer mới
        self.debounce_timer = self.root.after(delay, func)
    
    # Các phương thức xử lý ảnh - Transformations
    # Hàm Áp dụng biến đổi âm bản
    def apply_negative(self):
        if not self.original_image:
            return
        if self.negative_var.get():
            self.active_transformation = 'negative'
            self.current_image = apply_negative(self.original_image)
        else:   
            self.active_transformation = None
            self.current_image = self.original_image.copy()
        self.display_images()
    
    # Hàm Áp dụng biến đổi logarit
    def apply_log(self):
        if not self.original_image:
            return
        c_factor = self.log_c_var.get()
        self.current_image = apply_log_transform(self.original_image, c_factor)
        self.display_images()
    
    # Hàm Cập nhật hệ số C khi slider thay đổi
    def on_log_change(self, value):
        self.log_value_label.config(text=f"{self.log_c_var.get():.2f}")
        self.active_transformation = 'log'
        # Sử dụng debouncing để tránh gọi quá nhiều lần
        self._debounce(self.apply_log, delay=200)
    
    # Hàm Áp dụng biến đổi tuyến tính từng phần
    def apply_piecewise_linear(self):
        if not self.original_image:
            return
        high_factor = self.piecewise_high_var.get()
        low_factor = self.piecewise_low_var.get()
        self.current_image = apply_piecewise_linear(self.original_image, high_factor, low_factor)
        self.display_images()
    
    # Hàm Cập nhật các tham số khi slider thay đổi
    def on_piecewise_change(self, value):
        self.piecewise_high_label.config(text=f"{self.piecewise_high_var.get():.2f}")
        self.piecewise_low_label.config(text=f"{self.piecewise_low_var.get():.2f}")
        self.active_transformation = 'piecewise'
        # Sử dụng debouncing để tránh gọi quá nhiều lần
        self._debounce(self.apply_piecewise_linear, delay=200)
    
    # Hàm Áp dụng biến đổi gamma
    def apply_gamma(self):
        if not self.original_image:
            return
        gamma = self.gamma_var.get()
        c_factor = self.gamma_c_var.get()
        self.current_image = apply_gamma_transform(self.original_image, gamma, c_factor)
        self.display_images()
    
    # Hàm Cập nhật các tham số khi slider thay đổi
    def on_gamma_change(self, value):
        self.gamma_c_label.config(text=f"{self.gamma_c_var.get():.2f}")
        self.gamma_label.config(text=f"{self.gamma_var.get():.2f}")
        self.active_transformation = 'gamma'
        # Sử dụng debouncing để tránh gọi quá nhiều lần
        self._debounce(self.apply_gamma, delay=200)
    
    # Các phương thức xử lý ảnh - Filters
    # Hàm Áp dụng bộ lọc trung bình
    def apply_average_filter(self):
        if not self.original_image:
            return
        start_time = time.time()
        size = int(self.avg_size_var.get())
        if size % 2 == 0:
            size += 1
        img_array = np.array(self.original_image.convert('RGB'))
        k = np.ones((size, size)) / (size * size)
        img_filtered = Conv(img_array, k)
        self.current_image = Image.fromarray(img_filtered)
        total_time = (time.time() - start_time) * 1000  # Chuyển sang ms
        self.active_transformation = 'average'
        self._update_spatial_timing_info("Lọc Trung Bình", f"Kích thước={size}", total_time)
        self.display_images()
    
    # Hàm Cập nhật các tham số khi slider thay đổi
    def on_avg_change(self, value):
        size = int(self.avg_size_var.get())
        if size % 2 == 0:
            size += 1
        self.avg_size_label.config(text=str(size))
        self.active_transformation = 'average'
        # Sử dụng debouncing để tránh gọi quá nhiều lần
        self._debounce(self.apply_average_filter, delay=200)
    
    # Hàm Áp dụng bộ lọc Gaussian
    def apply_gaussian_filter(self):
        if not self.original_image:
            return
        start_time = time.time()
        sigma = self.gauss_sigma_var.get()
        size = int(self.gauss_size_var.get())
        if size % 2 == 0:
            size += 1
        img_array = np.array(self.original_image.convert('RGB'))
        k = Gausskernel(size, sigma)
        img_filtered = Conv(img_array, k)
        self.current_image = Image.fromarray(img_filtered)
        total_time = (time.time() - start_time) * 1000  # Chuyển sang ms
        self.active_transformation = 'gaussian'
        self._update_spatial_timing_info("Lọc Gaussian", f"Kích thước={size}, Sigma={sigma:.2f}", total_time)
        self.display_images()
    
    # Hàm Cập nhật các tham số khi slider thay đổi
    def on_gauss_change(self, value):
        size = int(self.gauss_size_var.get())
        if size % 2 == 0:
            size += 1
        self.gauss_size_label.config(text=str(size))
        self.gauss_sigma_label.config(text=f"{self.gauss_sigma_var.get():.2f}")
        self.active_transformation = 'gaussian'
        # Sử dụng debouncing để tránh gọi quá nhiều lần
        self._debounce(self.apply_gaussian_filter, delay=200)
    
    # Hàm Áp dụng bộ lọc trung vị
    def apply_median_filter(self):
        if not self.original_image:
            return
        start_time = time.time()
        size = int(self.median_size_var.get())
        if size % 2 == 0:
            size += 1
        img_array = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        img_filtered = apply_neighborhood_filter(img_array, size, 'median')
        self.current_image = Image.fromarray(img_filtered)
        total_time = (time.time() - start_time) * 1000  # Chuyển sang ms
        self.active_transformation = 'median'
        self._update_spatial_timing_info("Lọc Trung Vị", f"Kích thước={size}", total_time)
        self.display_images()
    
    # Hàm Cập nhật các tham số khi slider thay đổi
    def on_median_change(self, value):
        size = int(self.median_size_var.get())
        if size % 2 == 0:
            size += 1
        self.median_size_label.config(text=str(size))
        self.active_transformation = 'median'
        # Sử dụng debouncing để tránh gọi quá nhiều lần
        self._debounce(self.apply_median_filter, delay=200)
    
    # Hàm Áp dụng bộ lọc Max
    def apply_max_filter(self):
        if not self.original_image:
            return
        start_time = time.time()
        size = int(self.max_size_var.get())
        if size % 2 == 0:
            size += 1
        img_array = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        img_filtered = apply_neighborhood_filter(img_array, size, 'max')
        self.current_image = Image.fromarray(img_filtered)
        total_time = (time.time() - start_time) * 1000  # Chuyển sang ms
        self.active_transformation = 'max'
        self._update_spatial_timing_info("Lọc Max", f"Kích thước={size}", total_time)
        self.display_images()
    
    # Hàm Cập nhật các tham số khi slider thay đổi
    def on_max_change(self, value):
        size = int(self.max_size_var.get())
        if size % 2 == 0:
            size += 1
        self.max_size_label.config(text=str(size))
        self.active_transformation = 'max'
        # Sử dụng debouncing để tránh gọi quá nhiều lần
        self._debounce(self.apply_max_filter, delay=200)
    
    # Hàm Áp dụng bộ lọc Min
    def apply_min_filter(self):
        if not self.original_image:
            return
        start_time = time.time()
        size = int(self.min_size_var.get())
        if size % 2 == 0:
            size += 1
        img_array = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        img_filtered = apply_neighborhood_filter(img_array, size, 'min')
        self.current_image = Image.fromarray(img_filtered)
        total_time = (time.time() - start_time) * 1000  # Chuyển sang ms
        self.active_transformation = 'min'
        self._update_spatial_timing_info("Lọc Min", f"Kích thước={size}", total_time)
        self.display_images()
    
    # Hàm Cập nhật các tham số khi slider thay đổi
    def on_min_change(self, value):
        size = int(self.min_size_var.get())
        if size % 2 == 0:
            size += 1
        self.min_size_label.config(text=str(size))
        self.active_transformation = 'min'
        # Sử dụng debouncing để tránh gọi quá nhiều lần
        self._debounce(self.apply_min_filter, delay=200)
    
    # Hàm Áp dụng bộ lọc Midpoint
    def apply_midpoint_filter(self):
        if not self.original_image:
            return
        start_time = time.time()
        size = int(self.midpoint_size_var.get())
        if size % 2 == 0:
            size += 1
        img_array = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        img_filtered = apply_neighborhood_filter(img_array, size, 'midpoint')
        self.current_image = Image.fromarray(img_filtered)
        total_time = (time.time() - start_time) * 1000  # Chuyển sang ms
        self.active_transformation = 'midpoint'
        self._update_spatial_timing_info("Lọc Midpoint", f"Kích thước={size}", total_time)
        self.display_images()
    
    # Hàm Cập nhật các tham số khi slider thay đổi
    def on_midpoint_change(self, value):
        size = int(self.midpoint_size_var.get())
        if size % 2 == 0:
            size += 1
        self.midpoint_size_label.config(text=str(size))
        self.active_transformation = 'midpoint'
        # Sử dụng debouncing để tránh gọi quá nhiều lần
        self._debounce(self.apply_midpoint_filter, delay=200)
    
    # Hàm Áp dụng cân bằng histogram
    def apply_histogram_equalization(self):
        if not self.original_image:
            return
        brightness_factor = self.hist_value_var.get()
        self.current_image = apply_histogram_equalization(self.original_image, brightness_factor)
        self.active_transformation = 'histogram'
        self.display_images()
    
    # Hàm Cập nhật các tham số khi slider thay đổi
    def on_hist_change(self, value):
        self.hist_value_label.config(text=f"{self.hist_value_var.get():.2f}")
        self.active_transformation = 'histogram'
        # Sử dụng debouncing để tránh gọi quá nhiều lần
        self._debounce(self.apply_histogram_equalization, delay=200)
    
    # Các phương thức xử lý ảnh - Frequency Domain Filters
    # Hàm Áp dụng lọc thông thấp lý tưởng
    def apply_ideal_lowpass_filter(self):
        if not self.original_image:
            return
        D0 = self.ideal_lp_d0_var.get()
        self.current_image, timing_info = apply_ideal_lowpass_filter(self.original_image, D0)
        self.active_transformation = 'ideal_lowpass'
        self._update_timing_info(timing_info)
        self.display_images()
    
    def on_ideal_lp_change(self, value):
        self.ideal_lp_d0_label.config(text=f"{int(self.ideal_lp_d0_var.get())}")
        self.active_transformation = 'ideal_lowpass'
        self._debounce(self.apply_ideal_lowpass_filter, delay=200)
    
    # Hàm Áp dụng lọc thông thấp Gaussian
    def apply_gaussian_lowpass_filter(self):
        if not self.original_image:
            return
        D0 = self.gauss_lp_d0_var.get()
        self.current_image, timing_info = apply_gaussian_lowpass_filter(self.original_image, D0)
        self.active_transformation = 'gaussian_lowpass'
        self._update_timing_info(timing_info)
        self.display_images()
    
    def on_gauss_lp_change(self, value):
        self.gauss_lp_d0_label.config(text=f"{int(self.gauss_lp_d0_var.get())}")
        self.active_transformation = 'gaussian_lowpass'
        self._debounce(self.apply_gaussian_lowpass_filter, delay=200)
    
    # Hàm Áp dụng lọc thông thấp Butterworth
    def apply_butterworth_lowpass_filter(self):
        if not self.original_image:
            return
        D0 = self.butter_lp_d0_var.get()
        n = int(self.butter_lp_n_var.get())
        self.current_image, timing_info = apply_butterworth_lowpass_filter(self.original_image, D0, n)
        self.active_transformation = 'butterworth_lowpass'
        self._update_timing_info(timing_info)
        self.display_images()
    
    def on_butter_lp_change(self, value):
        self.butter_lp_d0_label.config(text=f"{int(self.butter_lp_d0_var.get())}")
        self.butter_lp_n_label.config(text=f"{int(self.butter_lp_n_var.get())}")
        self.active_transformation = 'butterworth_lowpass'
        self._debounce(self.apply_butterworth_lowpass_filter, delay=200)
    
    # Hàm Áp dụng lọc thông cao lý tưởng
    def apply_ideal_highpass_filter(self):
        if not self.original_image:
            return
        D0 = self.ideal_hp_d0_var.get()
        self.current_image, timing_info = apply_ideal_highpass_filter(self.original_image, D0)
        self.active_transformation = 'ideal_highpass'
        self._update_timing_info(timing_info)
        self.display_images()
    
    def on_ideal_hp_change(self, value):
        self.ideal_hp_d0_label.config(text=f"{int(self.ideal_hp_d0_var.get())}")
        self.active_transformation = 'ideal_highpass'
        self._debounce(self.apply_ideal_highpass_filter, delay=200)
    
    # Hàm Áp dụng lọc thông cao Gaussian
    def apply_gaussian_highpass_filter(self):
        if not self.original_image:
            return
        D0 = self.gauss_hp_d0_var.get()
        self.current_image, timing_info = apply_gaussian_highpass_filter(self.original_image, D0)
        self.active_transformation = 'gaussian_highpass'
        self._update_timing_info(timing_info)
        self.display_images()
    
    def on_gauss_hp_change(self, value):
        self.gauss_hp_d0_label.config(text=f"{int(self.gauss_hp_d0_var.get())}")
        self.active_transformation = 'gaussian_highpass'
        self._debounce(self.apply_gaussian_highpass_filter, delay=200)
    
    # Hàm Áp dụng lọc thông cao Butterworth
    def apply_butterworth_highpass_filter(self):
        if not self.original_image:
            return
        D0 = self.butter_hp_d0_var.get()
        n = int(self.butter_hp_n_var.get())
        self.current_image, timing_info = apply_butterworth_highpass_filter(self.original_image, D0, n)
        self.active_transformation = 'butterworth_highpass'
        self._update_timing_info(timing_info)
        self.display_images()
    
    def on_butter_hp_change(self, value):
        self.butter_hp_d0_label.config(text=f"{int(self.butter_hp_d0_var.get())}")
        self.butter_hp_n_label.config(text=f"{int(self.butter_hp_n_var.get())}")
        self.active_transformation = 'butterworth_highpass'
        self._debounce(self.apply_butterworth_highpass_filter, delay=200)
    
    # Hàm Reset ảnh về ban đầu
    def reset_image(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.active_transformation = None
            self._reset_all_values()
            self.display_images()
    
    # Hàm Áp dụng thay đổi lên ảnh gốc
    def apply_to_original(self):
        if self.current_image and self.original_image:
            self.original_image = self.current_image.copy()
            self.display_images()
            messagebox.showinfo("Thông báo", "Đã áp dụng thay đổi lên ảnh gốc.")
    
    # Hàm Lưu ảnh
    def save_image(self):
        if not self.current_image:
            messagebox.showwarning("Cảnh báo", "Không có ảnh để lưu.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if file_path:
            self.current_image.save(file_path)
            messagebox.showinfo("Thông báo", "Đã lưu ảnh thành công.")
    
    # Hàm So sánh thời gian giữa Gaussian miền không gian và miền tần số
    def compare_gaussian_filters(self):
        if not self.original_image:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước khi so sánh.")
            return
        
        # Lấy tham số từ GUI
        sigma = self.gauss_sigma_var.get()
        size = int(self.gauss_size_var.get())
        if size % 2 == 0:
            size += 1
        
        # Tính D0 tương đương (có thể dùng công thức D0 ≈ 2*sigma hoặc để người dùng chọn)
        # Ở đây dùng D0 từ frequency filter nếu có, nếu không thì tính từ sigma
        D0 = self.gauss_lp_d0_var.get()
        
        # Tạo dialog để hiển thị kết quả
        result_window = tk.Toplevel(self.root)
        result_window.title("So Sánh Thời Gian - Gaussian Filter")
        result_window.geometry("600x500")
        result_window.configure(bg='#f0f2f5')
        result_window.transient(self.root)
        result_window.grab_set()
        
        # Frame tiêu đề
        title_frame = tk.Frame(result_window, bg='#17a2b8', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="⚖️ So Sánh Thời Gian Xử Lý", 
                              bg='#17a2b8', fg='#ffffff', 
                              font=('Segoe UI', 14, 'bold'))
        title_label.pack(expand=True)
        
        # Frame nội dung
        content_frame = tk.Frame(result_window, bg='#f0f2f5')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Label thông báo đang xử lý
        processing_label = tk.Label(content_frame, 
                                   text="Đang xử lý... Vui lòng đợi...", 
                                   bg='#f0f2f5', fg='#212529', 
                                   font=('Segoe UI', 11))
        processing_label.pack(pady=20)
        
        result_window.update()
        
        # Chạy so sánh trong thread riêng để không làm đóng băng GUI
        def run_comparison():
            try:
                # 1. Lọc Gaussian miền không gian
                start_spatial = time.time()
                img_array = np.array(self.original_image.convert('RGB'))
                k = Gausskernel(size, sigma)
                img_filtered_spatial = Conv(img_array, k)
                time_spatial = (time.time() - start_spatial) * 1000  # ms
                
                # 2. Lọc Gaussian miền tần số
                start_frequency = time.time()
                _, timing_info_freq = apply_gaussian_lowpass_filter(self.original_image, D0)
                time_frequency = timing_info_freq['total_time']  # đã là ms
                
                # Tính phần trăm chênh lệch
                if time_spatial > 0:
                    diff_percent = ((time_frequency - time_spatial) / time_spatial) * 100
                else:
                    diff_percent = 0
                
                # Cập nhật GUI với kết quả
                self.root.after(0, lambda: self._display_comparison_result(
                    result_window, content_frame, processing_label,
                    size, sigma, D0, time_spatial, time_frequency, diff_percent
                ))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi khi so sánh: {str(e)}"))
                result_window.destroy()
        
        comparison_thread = threading.Thread(target=run_comparison, daemon=True)
        comparison_thread.start()
    
    # Hiển thị kết quả so sánh
    def _display_comparison_result(self, window, content_frame, processing_label,
                                   size, sigma, D0, time_spatial, time_frequency, diff_percent):
        # Xóa label đang xử lý
        processing_label.destroy()
        
        # Frame tham số
        params_frame = tk.Frame(content_frame, bg='#ffffff', relief='solid', bd=1)
        params_frame.pack(fill=tk.X, pady=(0, 15))
        
        params_title = tk.Label(params_frame, text="📋 Tham Số Sử Dụng", 
                               bg='#ffffff', fg='#17a2b8', 
                               font=('Segoe UI', 11, 'bold'))
        params_title.pack(pady=(10, 5))
        
        params_text = f"Miền Không Gian: Kích thước={size}, Sigma={sigma:.2f}\n"
        params_text += f"Miền Tần Số: D0={D0:.1f}"
        params_label = tk.Label(params_frame, text=params_text, 
                               bg='#ffffff', fg='#212529', 
                               font=('Segoe UI', 10), justify='left')
        params_label.pack(pady=(0, 10), padx=15)
        
        # Frame kết quả miền không gian
        spatial_frame = tk.Frame(content_frame, bg='#e7f3ff', relief='solid', bd=2)
        spatial_frame.pack(fill=tk.X, pady=(0, 10))
        
        spatial_title = tk.Label(spatial_frame, text="🌐 Miền Không Gian", 
                                bg='#e7f3ff', fg='#0056b3', 
                                font=('Segoe UI', 12, 'bold'))
        spatial_title.pack(pady=(10, 5))
        
        spatial_time = tk.Label(spatial_frame, 
                               text=f"Thời gian: {time_spatial:.3f} ms", 
                               bg='#e7f3ff', fg='#212529', 
                               font=('Segoe UI', 11))
        spatial_time.pack(pady=(0, 10))
        
        # Frame kết quả miền tần số
        frequency_frame = tk.Frame(content_frame, bg='#fff3cd', relief='solid', bd=2)
        frequency_frame.pack(fill=tk.X, pady=(0, 10))
        
        freq_title = tk.Label(frequency_frame, text="📊 Miền Tần Số", 
                             bg='#fff3cd', fg='#856404', 
                             font=('Segoe UI', 12, 'bold'))
        freq_title.pack(pady=(10, 5))
        
        freq_time = tk.Label(frequency_frame, 
                            text=f"Thời gian: {time_frequency:.3f} ms", 
                            bg='#fff3cd', fg='#212529', 
                            font=('Segoe UI', 11))
        freq_time.pack(pady=(0, 10))
        
        # Frame so sánh
        compare_frame = tk.Frame(content_frame, bg='#d1ecf1', relief='solid', bd=2)
        compare_frame.pack(fill=tk.X, pady=(0, 10))
        
        compare_title = tk.Label(compare_frame, text="⚖️ Kết Quả So Sánh", 
                                bg='#d1ecf1', fg='#0c5460', 
                                font=('Segoe UI', 12, 'bold'))
        compare_title.pack(pady=(10, 5))
        
        if time_spatial < time_frequency:
            winner = "Miền Không Gian"
            color = '#28a745'
        elif time_frequency < time_spatial:
            winner = "Miền Tần Số"
            color = '#ffc107'
        else:
            winner = "Bằng nhau"
            color = '#6c757d'
        
        winner_label = tk.Label(compare_frame, 
                               text=f"⚡ Nhanh hơn: {winner}", 
                               bg='#d1ecf1', fg=color, 
                               font=('Segoe UI', 11, 'bold'))
        winner_label.pack(pady=(0, 5))
        
        diff_text = f"Chênh lệch: {abs(diff_percent):.2f}%"
        if diff_percent > 0:
            diff_text += " (Miền Tần Số chậm hơn)"
        elif diff_percent < 0:
            diff_text += " (Miền Tần Số nhanh hơn)"
        else:
            diff_text += " (Bằng nhau)"
        
        diff_label = tk.Label(compare_frame, text=diff_text, 
                             bg='#d1ecf1', fg='#212529', 
                             font=('Segoe UI', 10))
        diff_label.pack(pady=(0, 10))
        
        # Nút đóng
        close_btn = tk.Button(content_frame, text="Đóng", 
                             command=window.destroy,
                             bg='#6c757d', fg='#ffffff', 
                             font=('Segoe UI', 10, 'bold'),
                             relief='raised', bd=2, cursor='hand2', 
                             padx=30, pady=8)
        close_btn.pack(pady=(10, 0))

