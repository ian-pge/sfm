import os
import sys
import argparse
import re
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk

# Configuration
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
LABELS = {
    'fl': '_fl',
    'fr': '_fr',
    'bl': '_bl',
    'br': '_br'
}
LABEL_SUFFIXES = list(LABELS.values())

class ImageLabeler:
    def __init__(self, root, folder):
        self.root = root
        self.folder = Path(folder)
        self.images = self.load_images(self.folder)
        self.index = 0
        
        # Theme
        self.bg_color = "#1e1e1e"
        self.fg_color = "#e0e0e0"
        self.btn_bg = "#333333"
        self.btn_fg = "#ffffff"
        self.accent_color = "#4a90e2"
        
        self.root.configure(bg=self.bg_color)
        self.root.title("Car Image Labeler")
        self.root.geometry("1024x768")
        
        # Layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Main Container
        self.main_container = tk.Frame(root, bg=self.bg_color)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left: Image Canvas
        self.canvas_frame = tk.Frame(self.main_container, bg=self.bg_color)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Label(self.canvas_frame, bg="#000000")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right: Controls
        self.controls_frame = tk.Frame(self.main_container, bg=self.bg_color, width=300)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        
        # Info
        self.lbl_info = tk.Label(self.controls_frame, text="", bg=self.bg_color, fg=self.fg_color, font=("Helvetica", 12))
        self.lbl_info.pack(pady=(0, 20), anchor="w")

        self.lbl_filename = tk.Label(self.controls_frame, text="", bg=self.bg_color, fg=self.fg_color, font=("Helvetica", 10), wraplength=280, justify="left")
        self.lbl_filename.pack(pady=(0, 20), anchor="w")
        
        # Grid Label
        tk.Label(self.controls_frame, text="Select Position:", bg=self.bg_color, fg=self.fg_color, font=("Helvetica", 12, "bold")).pack(pady=(0, 10), anchor="w")
        
        # 4-Square Grid Frame
        self.grid_frame = tk.Frame(self.controls_frame, bg=self.bg_color)
        self.grid_frame.pack()
        
        btn_size = 10
        btn_font = ("Helvetica", 14, "bold")
        
        # Buttons
        self.btn_fl = tk.Button(self.grid_frame, text="FL", command=lambda: self.label_image('fl'), 
                              bg=self.btn_bg, fg=self.btn_fg, font=btn_font, width=6, height=3)
        self.btn_fr = tk.Button(self.grid_frame, text="FR", command=lambda: self.label_image('fr'), 
                              bg=self.btn_bg, fg=self.btn_fg, font=btn_font, width=6, height=3)
        self.btn_bl = tk.Button(self.grid_frame, text="BL", command=lambda: self.label_image('bl'), 
                              bg=self.btn_bg, fg=self.btn_fg, font=btn_font, width=6, height=3)
        self.btn_br = tk.Button(self.grid_frame, text="BR", command=lambda: self.label_image('br'), 
                              bg=self.btn_bg, fg=self.btn_fg, font=btn_font, width=6, height=3)
        
        self.btn_fl.grid(row=0, column=0, padx=2, pady=2)
        self.btn_fr.grid(row=0, column=1, padx=2, pady=2)
        self.btn_bl.grid(row=1, column=0, padx=2, pady=2)
        self.btn_br.grid(row=1, column=1, padx=2, pady=2)
        
        # Navigation
        self.nav_frame = tk.Frame(self.controls_frame, bg=self.bg_color)
        self.nav_frame.pack(pady=40, fill=tk.X)
        
        self.btn_skip = tk.Button(self.nav_frame, text="Skip / Next", command=self.next_image,
                                bg=self.btn_bg, fg=self.btn_fg)
        self.btn_skip.pack(fill=tk.X, pady=5)

        self.root.update_idletasks() # Force update to get geometry info
        self.show_image()

    def load_images(self, folder):
        files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in EXTENSIONS])
        return files

    def show_image(self):
        if not self.images:
            self.lbl_info.config(text="No images found.")
            return

        if self.index >= len(self.images):
            self.lbl_info.config(text="Finished!")
            self.lbl_filename.config(text="")
            self.canvas.config(image='', text="No more images")
            return
            
        current_file = self.images[self.index]
        
        # Check if file still exists (might be renamed externally or previous run)
        if not current_file.exists():
            # Try to find if it was renamed? Or just skip
            self.lbl_info.config(text=f"File missing: {current_file.name}")
            return

        self.lbl_info.config(text=f"Image {self.index + 1} / {len(self.images)}")
        self.lbl_filename.config(text=current_file.name)
        
        try:
            img = Image.open(current_file)
            
            # Resize logic
            # Get canvas size (or maximize within limit)
            # Since canvas might not have size yet on first run, wait for window to be drawn?
            # We used update_idletasks() in init, so might have some size.
            
            canv_w = self.canvas_frame.winfo_width()
            canv_h = self.canvas_frame.winfo_height()
            
            # Fallback if window not fully realized
            if canv_w <= 1: canv_w = 800
            if canv_h <= 1: canv_h = 600
            
            # Padding
            canv_w -= 20
            canv_h -= 20
            
            img_ratio = img.width / img.height
            canv_ratio = canv_w / canv_h
            
            if img_ratio > canv_ratio:
                new_w = canv_w
                new_h = int(canv_w / img_ratio)
            else:
                new_h = canv_h
                new_w = int(canv_h * img_ratio)
                
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.config(image=self.tk_img)
            
        except Exception as e:
            print(f"Error loading image: {e}")
            self.lbl_filename.config(text=f"Error: {e}")

    def label_image(self, position):
        if self.index >= len(self.images):
            return

        current_file = self.images[self.index]
        suffix = LABELS[position]
        
        stem = current_file.stem
        # Check if stems ends with any known suffix
        for s in LABEL_SUFFIXES:
            if stem.endswith(s):
                stem = stem[:-len(s)]
                break
        
        new_name = f"{stem}{suffix}{current_file.suffix}"
        new_path = current_file.parent / new_name
        
        try:
            current_file.rename(new_path)
            print(f"Renamed: {current_file.name} -> {new_name}")
            
            # For correctness, update the list at current index so next time we know what it is
            self.images[self.index] = new_path
            
            self.index += 1
            self.show_image()
            
        except OSError as e:
            messagebox.showerror("Error", f"Failed to rename file:\n{e}")

    def next_image(self):
        self.index += 1
        self.show_image()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label images with _fl, _fr, _bl, _br suffixes.")
    parser.add_argument("folder", help="Input folder containing images")
    args = parser.parse_args()
    
    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"Error: {folder} is not a valid directory")
        sys.exit(1)

    root = tk.Tk()
    app = ImageLabeler(root, folder)
    root.mainloop()
