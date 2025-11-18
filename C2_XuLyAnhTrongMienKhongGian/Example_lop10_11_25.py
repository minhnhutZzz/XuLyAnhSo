import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
from scipy.signal import convolve2d


class ImageProcessorGUI:
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
        
        # Thiết lập style cho giao diện sáng với màu xanh dương nổi bật
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
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame chính
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Frame bên trái: Hiển thị ảnh với viền xanh dương và các nút chức năng nổi bật
        left_outer = tk.Frame(main_frame, bg=self.blue_color, highlightthickness=3, highlightbackground=self.blue_color, relief='solid')
        left_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        left_frame = tk.Frame(left_outer, bg='#ffffff')
        left_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tiêu đề với gradient-like effect (sử dụng label)
        title_frame = tk.Frame(left_frame, bg='#ffffff', height=50)
        title_frame.pack(fill=tk.X, pady=(5, 10))
        title_frame.pack_propagate(False)
        title_label = tk.Label(title_frame, text="Ảnh Gốc và Ảnh Chỉnh Sửa", 
                              bg='#ffffff', fg=self.blue_color, font=('Segoe UI', 12, 'bold'))
        title_label.pack(expand=True)
        
        # Sub-frame cho ảnh gốc và hiện tại
        images_subframe = tk.Frame(left_frame, bg='#ffffff')
        images_subframe.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Ảnh gốc (trên) với viền xanh dương nổi bật
        orig_title = tk.Label(images_subframe, text="Ảnh Gốc", bg='#ffffff', fg=self.blue_dark, font=('Segoe UI', 10, 'bold'))
        orig_title.pack(pady=(0, 5))
        self.orig_frame = tk.Frame(images_subframe, bg=self.blue_light, highlightthickness=2, highlightbackground=self.blue_color, relief='groove')
        self.orig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)
        self.original_label = tk.Label(self.orig_frame, text="Chưa chọn ảnh", relief=tk.FLAT, bg='#f8f9fa', fg='#6c757d', font=('Segoe UI', 10))
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Ảnh hiện tại (dưới) với viền xanh dương nổi bật
        curr_title = tk.Label(images_subframe, text="Ảnh Chỉnh Sửa", bg='#ffffff', fg=self.blue_dark, font=('Segoe UI', 10, 'bold'))
        curr_title.pack(pady=(0, 5))
        self.curr_frame = tk.Frame(images_subframe, bg=self.blue_light, highlightthickness=2, highlightbackground=self.blue_color, relief='groove')
        self.curr_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10)
        self.current_label = tk.Label(self.curr_frame, text="Chưa chỉnh sửa", relief=tk.FLAT, bg='#f8f9fa', fg='#6c757d', font=('Segoe UI', 10))
        self.current_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame bên phải: Chức năng chi tiết với notebook tabs để tổ chức tốt hơn
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Tiêu đề bên phải
        right_title = tk.Label(right_frame, text="Công Cụ Xử Lý Ảnh", 
                              bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 12, 'bold'))
        right_title.pack(pady=(0, 10))
        
        # Frame chứa các nút chức năng
        buttons_frame = tk.Frame(right_frame, bg='#f0f2f5')
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
    
    # Tab 1: Biến Đổi
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
        # Khoảng 0.1-2.0: hệ số nhân với giá trị c tự động tính
        # 1.0 = giá trị tối ưu, < 1.0 = tối hơn, > 1.0 = sáng hơn
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
        # Khoảng 0.1-2.0: hệ số nhân với giá trị c tự động tính
        # 1.0 = giá trị tối ưu, < 1.0 = tối hơn, > 1.0 = sáng hơn
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
    
    # Tab 2: Lọc Ảnh
    def setup_filter_tab(self, parent):
        # Tạo Canvas với Scrollbar để có thể cuộn
        # Frame chứa canvas và scrollbar
        canvas_frame = tk.Frame(parent, bg='#f0f2f5')
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
        
        # Bind mousewheel để cuộn (chỉ khi chuột ở trong canvas)
        def on_mousewheel(event):
            # Chỉ cuộn nếu chuột đang ở trong canvas
            if canvas.winfo_containing(event.x_root, event.y_root) == canvas:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # Bind cho canvas và scrollable_frame
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
        
        # Average filter
        avg_frame = self.create_section_frame(scrollable_frame, "Lọc Trung Bình", '#28a745')
        avg_content = tk.Frame(avg_frame, bg='#f0f2f5')
        avg_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(avg_content, text="Kích thước:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.avg_size_var = tk.IntVar(value=3)
        avg_size_scale = ttk.Scale(avg_content, from_=3, to=15, variable=self.avg_size_var, orient=tk.HORIZONTAL, command=self.on_avg_change, length=200)
        avg_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        avg_size_label = tk.Label(avg_content, text="3", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        avg_size_label.pack(side=tk.LEFT)
        self.avg_size_label = avg_size_label
        
        # Gaussian filter
        gauss_frame = self.create_section_frame(scrollable_frame, "Lọc Gaussian", '#ffc107')
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
        
        # Median filter
        median_frame = self.create_section_frame(scrollable_frame, "Lọc Trung Vị", '#fd7e14')
        median_content = tk.Frame(median_frame, bg='#f0f2f5')
        median_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(median_content, text="Kích thước:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.median_size_var = tk.IntVar(value=3)
        median_size_scale = ttk.Scale(median_content, from_=3, to=15, variable=self.median_size_var, orient=tk.HORIZONTAL, command=self.on_median_change, length=200)
        median_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        median_size_label = tk.Label(median_content, text="3", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        median_size_label.pack(side=tk.LEFT)
        self.median_size_label = median_size_label
        
        # Max filter
        max_frame = self.create_section_frame(scrollable_frame, "Lọc Max", '#dc3545')
        max_content = tk.Frame(max_frame, bg='#f0f2f5')
        max_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(max_content, text="Kích thước:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.max_size_var = tk.IntVar(value=3)
        max_size_scale = ttk.Scale(max_content, from_=3, to=15, variable=self.max_size_var, orient=tk.HORIZONTAL, command=self.on_max_change, length=200)
        max_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        max_size_label = tk.Label(max_content, text="3", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        max_size_label.pack(side=tk.LEFT)
        self.max_size_label = max_size_label
        
        # Min filter
        min_frame = self.create_section_frame(scrollable_frame, "Lọc Min", '#6f42c1')
        min_content = tk.Frame(min_frame, bg='#f0f2f5')
        min_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(min_content, text="Kích thước:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.min_size_var = tk.IntVar(value=3)
        min_size_scale = ttk.Scale(min_content, from_=3, to=15, variable=self.min_size_var, orient=tk.HORIZONTAL, command=self.on_min_change, length=200)
        min_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        min_size_label = tk.Label(min_content, text="3", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        min_size_label.pack(side=tk.LEFT)
        self.min_size_label = min_size_label
        
        # Midpoint filter
        midpoint_frame = self.create_section_frame(scrollable_frame, "Lọc Midpoint", '#20c997')
        midpoint_content = tk.Frame(midpoint_frame, bg='#f0f2f5')
        midpoint_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(midpoint_content, text="Kích thước:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.midpoint_size_var = tk.IntVar(value=3)
        midpoint_size_scale = ttk.Scale(midpoint_content, from_=3, to=15, variable=self.midpoint_size_var, orient=tk.HORIZONTAL, command=self.on_midpoint_change, length=200)
        midpoint_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))
        midpoint_size_label = tk.Label(midpoint_content, text="3", width=5, bg='#f0f2f5', fg=self.blue_color, font=('Segoe UI', 10))
        midpoint_size_label.pack(side=tk.LEFT)
        self.midpoint_size_label = midpoint_size_label
    
    # Tab 3: Khác
    def setup_other_tab(self, parent):
        # Histogram
        hist_frame = self.create_section_frame(parent, "Cân Bằng Histogram", '#6f42c1')
        hist_content = tk.Frame(hist_frame, bg='#f0f2f5')
        hist_content.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(hist_content, text="Độ sáng:", bg='#f0f2f5', fg='#212529', font=('Segoe UI', 10)).pack(side=tk.LEFT)
        self.hist_value_var = tk.DoubleVar(value=1.0)
        # Khoảng 0.1-2.0: hệ số nhân với ảnh sau khi cân bằng histogram
        # 1.0 = giữ nguyên, < 1.0 = tối hơn, > 1.0 = sáng hơn
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
    
    # Các phương thức xử lý ảnh giữ nguyên như trước
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
        if file_path:
            # Dùng cv2.imread() để đọc ảnh (giống các file mẫu)
            img_bgr = cv2.imread(file_path)
            if img_bgr is not None:
                # Chuyển BGR sang RGB (OpenCV dùng BGR, nhưng PIL và hiển thị dùng RGB)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # Chuyển numpy array sang PIL Image để tích hợp với Tkinter
                self.original_image = Image.fromarray(img_rgb)
                self.current_image = self.original_image.copy()
                self.active_transformation = None
            else:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh từ file này.")
                return
            # Reset các giá trị về mặc định
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
            self.display_images()
    
    # Resize ảnh giữ nguyên tỷ lệ để fit vào kích thước cho trước
    def resize_image_to_fit(self, image, max_width, max_height):
        if image is None:
            return None
        
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        
        # Tính toán kích thước mới giữ nguyên tỷ lệ
        if original_width > max_width or original_height > max_height:
            if max_width / max_height > aspect_ratio:
                # Fit theo chiều cao
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Fit theo chiều rộng
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
        else:
            # Ảnh nhỏ hơn frame, giữ nguyên kích thước
            new_width = original_width
            new_height = original_height
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Hiển thị ảnh
    def display_images(self):
        if self.original_image:
            # Đợi GUI được render để lấy kích thước frame chính xác
            self.root.update_idletasks()
            
            # Lấy kích thước frame thực tế
            max_width, max_height = self._get_display_size()
            
            # Resize ảnh gốc
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
            
            # Cập nhật lại sau khi frame được render hoàn toàn để đảm bảo kích thước chính xác
            self.root.after(50, self._refresh_images)
    
    # Lấy kích thước hiển thị cho ảnh dựa trên frame thực tế
    def _get_display_size(self):
        try:
            # Thử lấy kích thước thực tế của frame
            frame_width = self.orig_frame.winfo_width()
            frame_height = self.orig_frame.winfo_height()
            
            # Nếu frame đã được render (kích thước > 1)
            if frame_width > 30 and frame_height > 30:
                # Trừ padding (10px mỗi bên) và border (2px mỗi bên) = tổng 24px
                # Và thêm một chút margin an toàn
                display_width = frame_width - 40
                display_height = frame_height - 40
                return (max(100, display_width), max(100, display_height))
        except:
            pass
        
        # Fallback: Tính toán dựa trên kích thước cửa sổ
        try:
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            
            if window_width > 200 and window_height > 200:
                # Frame bên trái chiếm khoảng 55% chiều rộng
                # Mỗi ảnh chiếm khoảng 42% chiều cao (chia đôi cho 2 ảnh)
                estimated_width = int(window_width * 0.50) - 60  # Trừ các padding
                estimated_height = int((window_height - 200) * 0.42) - 40  # Trừ tiêu đề và padding
                return (max(300, estimated_width), max(200, estimated_height))
        except:
            pass
        
        # Kích thước mặc định an toàn (nhỏ hơn để đảm bảo không vượt ra ngoài)
        return (500, 350)
    
    # Làm mới kích thước ảnh sau khi frame được render hoàn toàn
    def _refresh_images(self):
        if self.original_image:
            max_width, max_height = self._get_display_size()
            
            # Resize lại ảnh với kích thước frame chính xác
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
                # Nếu có lỗi, bỏ qua và giữ nguyên ảnh hiện tại
                pass
    
    # Các phương thức xử lý ảnh
    # Áp dụng biến đổi âm bản
    def apply_negative(self):
        if not self.original_image:
            return
        if self.negative_var.get():
            self.active_transformation = 'negative'
            # Chuyển PIL Image sang numpy array (RGB) 
            img_rgb = np.array(self.original_image.convert('RGB'))
            
            # Áp dụng công thức âm bản
            img_neg = 255 - img_rgb
            
            # Tạo ảnh mới từ array
            self.current_image = Image.fromarray(img_neg, mode='RGB')
        else:   
            self.active_transformation = None
            self.current_image = self.original_image.copy()
        self.display_images()
    
    # Áp dụng biến đổi logarit
    def apply_log(self):
        if not self.original_image:
            return
        
        # Chuyển PIL Image sang numpy array RGB 
        img_rgb = np.array(self.original_image.convert('RGB'), dtype=np.float32)
        
        # Tính hệ số c tối ưu tự động
        max_value = np.max(img_rgb)
        c_optimal = 255 / np.log(1 + max_value)
        
        # Lấy hệ số từ slider (0.1-2.0) và nhân với c_optimal
        # slider = 1.0 → dùng c_optimal (tối ưu)
        # slider < 1.0 → c nhỏ hơn (ảnh tối hơn)
        # slider > 1.0 → c lớn hơn (ảnh sáng hơn)
        slider_value = self.log_c_var.get()
        c = slider_value * c_optimal
        
        # Áp dụng công thức log: S = c * log(1 + r)
        img_log = c * np.log(1 + img_rgb)
        
        # Chuyển về uint8 và clip về [0, 255]
        img_log = np.clip(img_log, 0, 255).astype(np.uint8)
        
        # Tạo ảnh mới từ array
        self.current_image = Image.fromarray(img_log, mode='RGB')
        self.display_images()
    
    # Cập nhật hệ số C khi slider thay đổi
    def on_log_change(self, value):
        self.log_value_label.config(text=f"{self.log_c_var.get():.2f}")
        self.active_transformation = 'log'
        self.apply_log()
    
    # Áp dụng biến đổi tuyến tính từng phần 
    def apply_piecewise_linear(self):
        if not self.original_image:
            return
        # Chuyển PIL Image sang numpy array RGB
        img_rgb = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        
        # Chuyển RGB sang HSV trực tiếp 
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        Ih, Is, Iv = cv2.split(hsv)
        
        # Lấy các tham số từ slider
        # a, b là các điểm breakpoint cố định
        a = 50
        b = 150
        # v, w là giá trị tại các điểm breakpoint, map từ slider
        # piecewise_low_var: 0.1-1.0 -> v: 20-200 (giá trị tại breakpoint a)
        # piecewise_high_var: 0.1-3.0 -> w: 10-300 (giá trị tại breakpoint b)
        v = int(self.piecewise_low_var.get() * 200)  # Scale từ 0.1-1.0 thành 20-200
        w = int(self.piecewise_high_var.get() * 100)  # Scale từ 0.1-3.0 thành 10-300
        
        # Hàm piecewise linear transformation (giống file mẫu)
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
        
        # Chuyển HSV về RGB trực tiếp (giống file mẫu)
        img_rgb_out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        
        # Tạo ảnh mới từ array
        self.current_image = Image.fromarray(img_rgb_out, mode='RGB')
        self.display_images()
    
    # Cập nhật các tham số khi slider thay đổi
    def on_piecewise_change(self, value):
        self.piecewise_high_label.config(text=f"{self.piecewise_high_var.get():.2f}")
        self.piecewise_low_label.config(text=f"{self.piecewise_low_var.get():.2f}")
        self.active_transformation = 'piecewise'
        self.apply_piecewise_linear()
    
    # Áp dụng biến đổi gamma
    def apply_gamma(self):
        if not self.original_image:
            return
        
        gamma = self.gamma_var.get()
        # Chuyển PIL Image sang numpy array RGB
        img_rgb = np.array(self.original_image.convert('RGB'), dtype=np.float32)
        
        # Tính hệ số c tối ưu tự động
        # Công thức: c_optimal = 255 / (max_value^gamma)
        max_value = np.max(img_rgb)
        c_optimal = 255 / (max_value ** gamma)
        
        # Lấy hệ số từ slider (0.1-2.0) và nhân với c_optimal
        # slider = 1.0 → dùng c_optimal (tối ưu)
        # slider < 1.0 → c nhỏ hơn (ảnh tối hơn)
        # slider > 1.0 → c lớn hơn (ảnh sáng hơn)
        slider_value = self.gamma_c_var.get()
        c = slider_value * c_optimal
        
        # Áp dụng công thức gamma: S = c * (r^gamma) (giống file mẫu)
        img_gamma = c * (img_rgb ** gamma)
        
        # Chuyển về uint8 và clip về [0, 255]
        img_gamma = np.clip(img_gamma, 0, 255).astype(np.uint8)

        # Tạo ảnh mới từ array
        self.current_image = Image.fromarray(img_gamma, mode='RGB')
        self.display_images()
    
    # Cập nhật các tham số khi slider thay đổi
    def on_gamma_change(self, value):
        self.gamma_c_label.config(text=f"{self.gamma_c_var.get():.2f}")
        self.gamma_label.config(text=f"{self.gamma_var.get():.2f}")
        self.active_transformation = 'gamma'
        self.apply_gamma()
    
    # Áp dụng bộ lọc trung bình
    def apply_average_filter(self):
        if not self.original_image:
            return
        size = int(self.avg_size_var.get())
        if size % 2 == 0:
            size += 1
        
        # Chuyển PIL Image sang numpy array RGB
        img_rgb = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        
        # Tạo mặt nạ lọc trung bình 
        k = np.ones((size, size)) / (size * size)
        
        # Tách các kênh màu
        r, g, b = cv2.split(img_rgb)
        
        # Lọc từng kênh bằng convolution 2D
        R = convolve2d(r, k, mode='same', boundary='symm')
        G = convolve2d(g, k, mode='same', boundary='symm')
        B = convolve2d(b, k, mode='same', boundary='symm')
        
        # Ghép lại ảnh sau khi lọc
        img_filtered = cv2.merge((np.uint8(np.clip(R, 0, 255)),
                                 np.uint8(np.clip(G, 0, 255)),
                                 np.uint8(np.clip(B, 0, 255))))
        
        # Tạo ảnh mới từ array
        self.current_image = Image.fromarray(img_filtered, mode='RGB')
        self.display_images()
    
    # Cập nhật các tham số khi slider thay đổi
    def on_avg_change(self, value):
        size = int(self.avg_size_var.get())
        if size % 2 == 0:
            size += 1
        self.avg_size_label.config(text=str(size))
        self.active_transformation = 'average'
        self.apply_average_filter()
    
    # Áp dụng bộ lọc Gaussian
    def apply_gaussian_filter(self):
        if not self.original_image:
            return
        
        # sigma: dộ lệch chuẩn
        # sigma càng nhỏ trọng số tâm càng cao, viền càng nhỏ
        # sigma càng lớn trọng số tâm giảm, viền gần bằng tâm
        sigma = self.gauss_sigma_var.get()
        size = int(self.gauss_size_var.get())
        if size % 2 == 0:
            size += 1
        
        # Chuyển PIL Image sang numpy array RGB
        img_rgb = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        
        # Chuyển RGB sang BGR cho OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Sử dụng cv2.GaussianBlur() để áp dụng Gaussian filter 
        filtered_bgr = cv2.GaussianBlur(img_bgr, (size, size), sigma)
        
        # Chuyển BGR về RGB
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        
        # Tạo ảnh mới từ array
        self.current_image = Image.fromarray(filtered_rgb, mode='RGB')
        self.display_images()
    
    # Cập nhật các tham số khi slider thay đổi
    def on_gauss_change(self, value):
        size = int(self.gauss_size_var.get())
        if size % 2 == 0:
            size += 1
        self.gauss_size_label.config(text=str(size))
        self.gauss_sigma_label.config(text=f"{self.gauss_sigma_var.get():.2f}")
        self.active_transformation = 'gaussian'
        self.apply_gaussian_filter()
    
    # Áp dụng bộ lọc trung vị
    def apply_median_filter(self):
        if not self.original_image:
            return
        size = int(self.median_size_var.get())
        if size % 2 == 0:
            size += 1
        
        # Chuyển PIL Image sang numpy array RGB
        img_rgb = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        
        # Chuyển RGB sang BGR cho OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Sử dụng cv2.medianBlur() để áp dụng median filter
        filtered_bgr = cv2.medianBlur(img_bgr, size)
        
        # Chuyển BGR về RGB
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        
        # Tạo ảnh mới từ array
        self.current_image = Image.fromarray(filtered_rgb, mode='RGB')
        self.display_images()
    
    # Cập nhật các tham số khi slider thay đổi
    def on_median_change(self, value):
        size = int(self.median_size_var.get())
        if size % 2 == 0:
            size += 1
        self.median_size_label.config(text=str(size))
        self.active_transformation = 'median'
        self.apply_median_filter()
    
    # Áp dụng bộ lọc Max
    def apply_max_filter(self):
        if not self.original_image:
            return
        size = int(self.max_size_var.get())
        if size % 2 == 0:
            size += 1
        
        # Chuyển PIL Image sang numpy array RGB
        img_rgb = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        
        # Chuyển RGB sang BGR cho OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Tạo kernel cho morphological operation
        kernel = np.ones((size, size), np.uint8)
        
        # Sử dụng cv2.dilate() để áp dụng max filter (dilation = max trong neighborhood)
        filtered_bgr = cv2.dilate(img_bgr, kernel, iterations=1)
        
        # Chuyển BGR về RGB
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        
        # Tạo ảnh mới từ array
        self.current_image = Image.fromarray(filtered_rgb, mode='RGB')
        self.display_images()
    
    # Cập nhật các tham số khi slider thay đổi
    def on_max_change(self, value):
        size = int(self.max_size_var.get())
        if size % 2 == 0:
            size += 1
        self.max_size_label.config(text=str(size))
        self.active_transformation = 'max'
        self.apply_max_filter()
    
    # Áp dụng bộ lọc Min
    def apply_min_filter(self):
        if not self.original_image:
            return
        size = int(self.min_size_var.get())
        if size % 2 == 0:
            size += 1
        
        # Chuyển PIL Image sang numpy array RGB
        img_rgb = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        
        # Chuyển RGB sang BGR cho OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Tạo kernel cho morphological operation
        kernel = np.ones((size, size), np.uint8)
        
        # Sử dụng cv2.erode() để áp dụng min filter (erosion = min trong neighborhood)
        filtered_bgr = cv2.erode(img_bgr, kernel, iterations=1)
        
        # Chuyển BGR về RGB
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        
        # Tạo ảnh mới từ array
        self.current_image = Image.fromarray(filtered_rgb, mode='RGB')
        self.display_images()
    
    # Cập nhật các tham số khi slider thay đổi
    def on_min_change(self, value):
        size = int(self.min_size_var.get())
        if size % 2 == 0:
            size += 1
        self.min_size_label.config(text=str(size))
        self.active_transformation = 'min'
        self.apply_min_filter()
    
    # Áp dụng bộ lọc Midpoint
    def apply_midpoint_filter(self):
        if not self.original_image:
            return
        size = int(self.midpoint_size_var.get())
        if size % 2 == 0:
            size += 1
        
        # Chuyển PIL Image sang numpy array RGB
        img_rgb = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        
        # Chuyển RGB sang BGR cho OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Tạo kernel cho morphological operation
        kernel = np.ones((size, size), np.uint8)
        
        # Tính max và min trong neighborhood
        max_img = cv2.dilate(img_bgr, kernel, iterations=1)
        min_img = cv2.erode(img_bgr, kernel, iterations=1)
        
        # Midpoint = (max + min) / 2
        filtered_bgr = ((max_img.astype(np.float32) + min_img.astype(np.float32)) / 2).astype(np.uint8)
        
        # Chuyển BGR về RGB
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        
        # Tạo ảnh mới từ array
        self.current_image = Image.fromarray(filtered_rgb, mode='RGB')
        self.display_images()
    
    # Cập nhật các tham số khi slider thay đổi
    def on_midpoint_change(self, value):
        size = int(self.midpoint_size_var.get())
        if size % 2 == 0:
            size += 1
        self.midpoint_size_label.config(text=str(size))
        self.active_transformation = 'midpoint'
        self.apply_midpoint_filter()
    
    # Áp dụng cân bằng histogram
    def apply_histogram_equalization(self):
        if not self.original_image:
            return
        
        # Chuyển PIL Image sang numpy array RGB
        img_rgb = np.array(self.original_image.convert('RGB'), dtype=np.uint8)
        
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
        
        # Áp dụng hệ số điều chỉnh độ sáng từ slider (0.1-2.0)
        # 1.0 = giữ nguyên, < 1.0 = tối hơn, > 1.0 = sáng hơn
        value = self.hist_value_var.get()
        if value != 1.0:
            equ_rgb = np.clip(equ_rgb.astype(np.float32) * value, 0, 255).astype(np.uint8)
        
        # Tạo ảnh mới từ array
        self.current_image = Image.fromarray(equ_rgb, mode='RGB')
        self.active_transformation = 'histogram'
        self.display_images()
    
    # Cập nhật các tham số khi slider thay đổi
    def on_hist_change(self, value):
        self.hist_value_label.config(text=f"{self.hist_value_var.get():.2f}")
        self.active_transformation = 'histogram'
        self.apply_histogram_equalization()
    
    # Reset ảnh về ban đầu
    def reset_image(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.negative_var.set(False)
            self.active_transformation = None
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
            self.display_images()
    
    # Áp dụng thay đổi lên ảnh gốc
    def apply_to_original(self):
        if self.current_image and self.original_image:
            self.original_image = self.current_image.copy()
            self.display_images()
            messagebox.showinfo("Thông báo", "Đã áp dụng thay đổi lên ảnh gốc.")
    

    # Lưu ảnh
    def save_image(self):
        if not self.current_image:
            messagebox.showwarning("Cảnh báo", "Không có ảnh để lưu.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if file_path:
            self.current_image.save(file_path)
            messagebox.showinfo("Thông báo", "Đã lưu ảnh thành công.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorGUI(root)
    root.mainloop()