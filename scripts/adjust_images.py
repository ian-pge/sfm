import argparse
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Improve resolution on high-DPI displays
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# Helper function for parallel processing (must be top-level)
def process_single_image(img_path, output_dir, brightness, contrast, saturation):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False
            
        # Apply adjustments
        adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        if saturation != 1.0:
            hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype("float32")
            hsv[..., 1] = hsv[..., 1] * saturation
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            adjusted = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
            
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), adjusted)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

class RoundedButton(tk.Canvas):
    def __init__(self, parent, width, height, corner_radius, padding=0, color="#FFFFFF", text_color="#000000", bg_color="#FFFFFF", text="", command=None):
        tk.Canvas.__init__(self, parent, borderwidth=0, relief="flat", highlightthickness=0, bg=bg_color)
        self.command = command
        self.width = width
        self.height = height
        self.corner_radius = corner_radius
        self.padding = padding
        self.color = color
        self.text_color = text_color
        self.text_str = text
        self.bg_color = bg_color

        if self.corner_radius > 0.5 * self.width:
            self.corner_radius = 0.5 * self.width
        if self.corner_radius > 0.5 * self.height:
            self.corner_radius = 0.5 * self.height

        self.configure(width=self.width + 2*padding, height=self.height + 2*padding)
        self._draw()
        
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_hover)
        self.bind("<Leave>", self._on_leave)

    def _draw(self):
        self.delete("all")
        # Draw rounded rect
        self.shape_id = self.create_rounded_rect(
            self.padding, self.padding, 
            self.width + self.padding, self.height + self.padding, 
            self.corner_radius, 
            fill=self.color, outline=self.color
        )
        # Draw text
        self.text_id = self.create_text(
            self.width/2 + self.padding, self.height/2 + self.padding, 
            text=self.text_str, fill=self.text_color, font=("Segoe UI" if sys.platform=="win32" else "Helvetica", 10, "bold")
        )

    def _on_press(self, event):
        # Flatten visual effect
        self.move(self.text_id, 1, 1)
        if self.command:
            self.command()

    def _on_release(self, event):
        self.move(self.text_id, -1, -1)

    def _on_hover(self, event):
        pass 

    def _on_leave(self, event):
        pass

    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x1 + radius, y1,
            x2 - radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.create_polygon(points, **kwargs, smooth=True)

class ImageAdjustmentApp:
    def __init__(self, root, img_paths, output_dir):
        self.root = root
        self.img_paths = img_paths
        self.output_dir = output_dir
        self.root.title("Image Adjustment Tool")
        self.root.geometry("1280x850")
        
        self.current_image_idx = 0
        
        # --- DARK THEME CONSTANTS ---
        self.bg_color = "#121212"       
        self.panel_color = "#1E1E1E"    
        self.text_main = "#E0E0E0"      
        self.text_sub = "#A0A0A0"       
        self.accent_color = "#BB86FC"   
        self.success_color = "#03DAC6"  
        self.slider_bg = "#333333"
        
        # Zoom state
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Images
        self.original_cv_image = None
        self.preview_cache_image = None # Downscaled version for fast updates
        self.current_display_image = None
        
        # Debounce
        self.last_update_request = 0
        self.update_pending = False
        
        self.setup_ui()
        
        # Start update loop
        self.check_updates()
        
        self.load_image(0)
        
    def setup_ui(self):
        self.root.configure(bg=self.bg_color)
        
        # Use simple fonts
        self.font_header = ("Helvetica", 14, "bold")
        self.font_label = ("Helvetica", 10)
        
        # --- LAYOUT ---
        self.root.columnconfigure(0, weight=1) 
        self.root.columnconfigure(1, weight=0, minsize=320)
        self.root.rowconfigure(0, weight=1)
        
        # LEFT: Canvas
        self.canvas_frame = tk.Frame(self.root, bg=self.bg_color)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#000000", highlightthickness=0, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind events for zoom/pan
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<MouseWheel>", self.on_zoom)        # Windows
        self.canvas.bind("<Button-4>", self.on_zoom)          # Linux Scroll Up
        self.canvas.bind("<Button-5>", self.on_zoom)          # Linux Scroll Down
        # Bind resize
        self.canvas.bind("<Configure>", self.on_resize)
        
        # Bottom Navigation
        self.nav_frame = tk.Frame(self.canvas_frame, bg=self.bg_color)
        self.nav_frame.pack(fill=tk.X, pady=(15, 0))
        
        nav_inner = tk.Frame(self.nav_frame, bg=self.bg_color)
        nav_inner.pack(anchor="center")
        
        self.btn_prev = RoundedButton(nav_inner, 100, 36, 18, color="#333333", text_color=self.text_main, bg_color=self.bg_color, text="PREV", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=10)
        
        self.lbl_counter = tk.Label(nav_inner, text="0 / 0", font=self.font_label, bg=self.bg_color, fg=self.text_sub)
        self.lbl_counter.pack(side=tk.LEFT, padx=15)
        
        self.btn_next = RoundedButton(nav_inner, 100, 36, 18, color="#333333", text_color=self.text_main, bg_color=self.bg_color, text="NEXT", command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=10)

        # RIGHT: Controls
        self.controls_frame = tk.Frame(self.root, bg=self.panel_color, padx=25, pady=30)
        self.controls_frame.grid(row=0, column=1, sticky="ns")
        
        tk.Label(self.controls_frame, text="Adjustments", font=self.font_header, bg=self.panel_color, fg=self.text_main).pack(anchor="w", pady=(0, 20))
        
        # Sliders
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Dark.Horizontal.TScale", background=self.panel_color, troughcolor="#121212", bordercolor=self.panel_color, lightcolor=self.panel_color, darkcolor=self.panel_color, sliderthickness=15)
        
        self.brightness_var = tk.DoubleVar(value=0)
        self.lbl_bri = self.add_slider("Brightness", self.brightness_var, -100, 100)
        
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.lbl_con = self.add_slider("Contrast", self.contrast_var, 0.0, 3.0)
        
        self.saturation_var = tk.DoubleVar(value=1.0)
        self.lbl_sat = self.add_slider("Saturation", self.saturation_var, 0.0, 3.0)
        
        # Reset
        tk.Frame(self.controls_frame, bg=self.panel_color, height=30).pack()
        self.btn_reset = RoundedButton(self.controls_frame, 250, 40, 20, color="#2C2C2C", text_color=self.text_sub, bg_color=self.panel_color, text="Reset Defaults", command=self.reset_defaults)
        self.btn_reset.pack(pady=5)
        
        # Save
        self.btn_save = RoundedButton(self.controls_frame, 250, 50, 25, color=self.success_color, text_color="#000000", bg_color=self.panel_color, text="Process All Images", command=self.start_processing)
        self.btn_save.pack(pady=(15, 0))
        
        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(self.controls_frame, variable=self.progress_var, maximum=100)
        self.lbl_status = tk.Label(self.controls_frame, text=f"Output: {self.output_dir.name}/", font=("Helvetica", 9), bg=self.panel_color, fg="#555555")
        self.lbl_status.pack(side=tk.BOTTOM, pady=10)

    def add_slider(self, label, var, min_v, max_v):
        wrapper = tk.Frame(self.controls_frame, bg=self.panel_color)
        wrapper.pack(fill=tk.X, pady=12)
        
        header = tk.Frame(wrapper, bg=self.panel_color)
        header.pack(fill=tk.X)
        tk.Label(header, text=label, font=("Helvetica", 10, "bold"), bg=self.panel_color, fg=self.text_main).pack(side=tk.LEFT)
        val_lbl = tk.Label(header, text="0.00", font=("Helvetica", 10), bg=self.panel_color, fg=self.accent_color)
        val_lbl.pack(side=tk.RIGHT)
        
        scale = ttk.Scale(wrapper, from_=min_v, to=max_v, orient=tk.HORIZONTAL, variable=var, command=lambda v: [self.request_update(), val_lbl.config(text=f"{var.get():.2f}")], style="Dark.Horizontal.TScale")
        scale.pack(fill=tk.X, pady=(6,0))
        return val_lbl

    # --- IMAGE HANDLING ---
    def prev_image(self):
        new_idx = self.current_image_idx - 1
        if new_idx < 0:
            new_idx = len(self.img_paths) - 1
        self.load_image(new_idx)
    
    def next_image(self):
        new_idx = self.current_image_idx + 1
        if new_idx >= len(self.img_paths):
            new_idx = 0
        self.load_image(new_idx)
        
    def load_image(self, idx):
        self.current_image_idx = idx
        path = str(self.img_paths[idx])
        self.original_cv_image = cv2.imread(path)
        
        if self.original_cv_image is None:
            print(f"Error loading {path}")
            return
            
        # Optimization: Create a cached downscaled version for preview logic
        # Max width 1920
        h, w = self.original_cv_image.shape[:2]
        max_dim = 1920
        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            self.preview_cache_image = cv2.resize(self.original_cv_image, (int(w*scale), int(h*scale)))
        else:
            self.preview_cache_image = self.original_cv_image.copy()
            
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        self.lbl_counter.config(text=f"{idx+1} / {len(self.img_paths)}")
        self.request_update(now=True)
        
    def check_updates(self):
        if self.update_pending and (time.time() - self.last_update_request > 0.05): # 50ms throttle
            self.do_update_preview()
            self.update_pending = False
            
        self.root.after(30, self.check_updates)
        
    def request_update(self, now=False):
        self.last_update_request = time.time()
        self.update_pending = True
        if now:
            self.do_update_preview()
            self.update_pending = False
        
    def do_update_preview(self):
        if self.preview_cache_image is None: return
        
        # Apply transforms on the CACHED image (likely downscaled)
        b = self.brightness_var.get()
        c = self.contrast_var.get()
        s = self.saturation_var.get()
        
        # 1. Processing
        adjusted = cv2.convertScaleAbs(self.preview_cache_image, alpha=c, beta=b)
        if s != 1.0:
            hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype("float32")
            hsv[..., 1] = hsv[..., 1] * s
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            adjusted = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
            
        # 2. Display Generation
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10: cw = 800
        if ch < 10: ch = 600
        
        h, w = adjusted.shape[:2]
        
        # Scale to window
        scale_w = cw / w
        scale_h = ch / h
        base_scale = min(scale_w, scale_h) * 0.9
        
        final_scale = base_scale * self.zoom_level
        new_w = int(w * final_scale)
        new_h = int(h * final_scale)
        
        if new_w <= 0 or new_h <= 0: return
        
        # Fast nearest neighbor for interaction if dragging? No, Linear is fast enough on 1080p
        resized = cv2.resize(adjusted, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb)
        self.current_display_image = ImageTk.PhotoImage(im_pil)
        
        self.canvas.delete("all")
        
        cx = cw // 2 + self.pan_x
        cy = ch // 2 + self.pan_y
        
        self.canvas.create_image(cx, cy, image=self.current_display_image, anchor=tk.CENTER)
        
        # self.canvas.create_text(10, 10, text=f"Zoom: {self.zoom_level:.1f}x", anchor="nw", fill="white")

    def reset_defaults(self):
        self.brightness_var.set(0)
        self.contrast_var.set(1.0)
        self.saturation_var.set(1.0)
        self.lbl_bri.config(text="0.00")
        self.lbl_con.config(text="1.00")
        self.lbl_sat.config(text="1.00")
        self.request_update(now=True)
        
    # --- ZOOM & PAN ---
    def on_zoom(self, event):
        if event.num == 4 or event.delta > 0:
            self.zoom_level *= 1.1
        elif event.num == 5 or event.delta < 0:
            self.zoom_level /= 1.1
            
        if self.zoom_level < 0.1: self.zoom_level = 0.1
        if self.zoom_level > 10.0: self.zoom_level = 10.0
            
        self.request_update(now=True)
        
    def on_drag_start(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
    def on_drag_motion(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.pan_x += dx
        self.pan_y += dy
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.request_update() # Allow throttle
        
    def on_resize(self, event):
        self.request_update()

    # --- PROCESSING ---
    def start_processing(self):
        # UI Feedback
        self.progress.pack(fill=tk.X, pady=10)
        self.btn_save.unbind("<ButtonPress-1>")
        
        b = self.brightness_var.get()
        c = self.contrast_var.get()
        s = self.saturation_var.get()
        
        threading.Thread(target=self.run_parallel, args=(b, c, s), daemon=True).start()
        
    def run_parallel(self, b, c, s):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        total = len(self.img_paths)
        completed = 0
        
        # Use simple os.cpu_count() workers
        workers = max(1, (sys.modules['os'].cpu_count() or 4) - 1)
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_image, p, self.output_dir, b, c, s): p 
                for p in self.img_paths
            }
            
            for future in as_completed(futures):
                completed += 1
                # Update UI safely
                self.root.after(0, lambda v=completed: self.progress_var.set((v/total) * 100))
                
        def on_done():
            messagebox.showinfo("Done", f"Saved to {self.output_dir}")
            self.root.destroy()
            
        self.root.after(0, on_done)


def main():
    parser = argparse.ArgumentParser(description="Image Adjustment GUI")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", help="Output directory")
    args = parser.parse_args()
    
    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        print(f"Error: {input_dir} not found")
        sys.exit(1)

    # --- AUTO-RENAME LOGIC ---
    # The user wants the input folder to be 'images_original'
    parent_dir = input_dir.parent
    desired_input_name = "images_original"
    desired_input_path = parent_dir / desired_input_name

    if input_dir.name != desired_input_name:
        if desired_input_path.exists():
            print(f"ℹ️  Target input folder '{desired_input_name}' already exists.")
            print(f"    Switching input from '{input_dir.name}' to '{desired_input_name}'")
            input_dir = desired_input_path
        else:
            print(f"🔄 Renaming input folder '{input_dir.name}' -> '{desired_input_name}'")
            try:
                input_dir.rename(desired_input_path)
                input_dir = desired_input_path
            except Exception as e:
                print(f"Error renaming folder: {e}")
                sys.exit(1)
    
    # -------------------------
        
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted([
        f for f in input_dir.glob("*")
        if f.suffix.lower() in valid_extensions and f.is_file()
    ])
    
    if not images:
        print(f"Error: No images found in {input_dir}")
        sys.exit(1)
        
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default output is sibling 'images' folder
        output_dir = input_dir.parent / "images"
        
    root = tk.Tk()
    app = ImageAdjustmentApp(root, images, output_dir)
    root.mainloop()

if __name__ == "__main__":
    main()
