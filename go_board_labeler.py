# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 17:11:03 2025

@author: Kevin
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import re

# Color mapping
COLOR_MAP = {0: "blue", 1: "white", 2: "black", 3: "red"}
LABEL_CHAR = {0: "N", 1: "B", 2: "W", 3: "H"}
CHAR_TO_LABEL = {"N": 0, "B": 1, "W": 2, "H": 3}

def custom_sort_key(file_path):
    """
    Custom sorting function for filenames like frame_00-00-18, frame_00-00-18-5
    Parse filename into numeric parts to ensure correct sorting
    """
    filename = os.path.basename(file_path)
    # Match frame_XX-XX-XX or frame_XX-XX-XX-Y format
    match = re.match(r'frame_(\d+)-(\d+)-(\d+)(?:-(\d+))?', filename)
    
    if match:
        # Extract numeric parts
        groups = match.groups()
        # Convert matched numbers to integers
        parts = [int(g) if g else 0 for g in groups]
        return parts
    
    # If not matching specific format, use regular natural sorting
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', filename)]

class GoBoardLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("19×19 Go Board Labeler")

        self.image_dir = filedialog.askdirectory(title="Select Image Folder")
        if not self.image_dir:  # If user cancels selection
            self.root.quit()
            return
            
        self.label_dir = filedialog.askdirectory(title="Select Label Folder")
        if not self.label_dir:  # If user cancels selection
            self.root.quit()
            return

        # Support multiple image formats
        self.image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif')
        
        # Get all supported image paths
        self.image_paths = []
        for ext in self.image_extensions:
            self.image_paths.extend(
                [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) 
                 if f.lower().endswith(ext)]
            )
            
        # Sort image paths using custom sorting function
        self.image_paths = sorted(self.image_paths, key=custom_sort_key)
        
        if not self.image_paths:
            messagebox.showinfo("No Images", "No compatible images found in the selected directory.")
            self.root.quit()
            return
        
        # Generate corresponding label paths for each image
        self.label_paths = []
        for img_path in self.image_paths:
            # Extract filename without extension
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Generate corresponding label file path
            label_path = os.path.join(self.label_dir, base_name + ".npy")
            self.label_paths.append(label_path)

        self.index = 0
        self.current_label = 1  # Default to B

        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.info_frame = tk.Frame(root)
        self.info_frame.pack(fill=tk.X)

        self.status_label = tk.Label(self.info_frame, text="Current Label: B")
        self.status_label.pack(side="left", padx=10)

        self.filename_label = tk.Label(self.info_frame, text="Image: ")
        self.filename_label.pack(side="left", padx=10)
        
        # Add image count display
        self.count_label = tk.Label(self.info_frame, text=f"Image 1/{len(self.image_paths)}")
        self.count_label.pack(side="left", padx=10)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(fill=tk.X)
        for i, ch in enumerate(["B", "W", "N", "H"]):
            tk.Button(self.button_frame, text=ch, command=lambda c=ch: self.set_label(c)).grid(row=0, column=i)
        tk.Button(self.button_frame, text="Clear Cell", command=self.clear_label).grid(row=0, column=4)
        tk.Button(self.button_frame, text="Save Label", command=self.save_label).grid(row=1, column=0, columnspan=2)
        tk.Button(self.button_frame, text="<< Prev", command=self.prev_image).grid(row=1, column=2)
        tk.Button(self.button_frame, text="Next >>", command=self.next_image).grid(row=1, column=3)

        # Add jump to specific image functionality
        self.jump_frame = tk.Frame(root)
        self.jump_frame.pack(pady=5, fill=tk.X)
        tk.Label(self.jump_frame, text="Jump to:").pack(side="left")
        self.jump_entry = tk.Entry(self.jump_frame, width=5)
        self.jump_entry.pack(side="left", padx=5)
        tk.Button(self.jump_frame, text="Go", command=self.jump_to_image).pack(side="left")

        # Button to show file list
        tk.Button(self.jump_frame, text="Show File List", command=self.show_file_list).pack(side="right", padx=10)

        self.load_image()

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

    def show_file_list(self):
        """Show file name list window"""
        file_list_window = tk.Toplevel(self.root)
        file_list_window.title("Image File List")
        
        # Set window size
        file_list_window.geometry("400x500")
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(file_list_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create listbox
        listbox = tk.Listbox(file_list_window, width=50, height=25, yscrollcommand=scrollbar.set)
        listbox.pack(fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        scrollbar.config(command=listbox.yview)
        
        # Fill file list
        for i, img_path in enumerate(self.image_paths):
            filename = os.path.basename(img_path)
            listbox.insert(tk.END, f"{i+1}. {filename}")
        
        # Double-click to jump to image
        listbox.bind("<Double-Button-1>", lambda e: self.jump_to_selected(listbox.curselection()))
        
        # Add explanation label
        tk.Label(file_list_window, text="Double-click an item to jump to that image").pack()

    def jump_to_selected(self, selection):
        """Jump to selected image in the list"""
        if selection:
            index = selection[0]
            self.save_label()  # Save current label
            self.index = index
            self.load_image()

    # Rest of the methods remain the same as in the Chinese version,
    # with comments and messages translated to English

    def load_image(self):
        if not self.image_paths:
            return
            
        try:
            img = Image.open(self.image_paths[self.index]).convert("RGB")
            img = img.resize((608, 608))
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.config(width=608, height=608)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

            if os.path.exists(self.label_paths[self.index]):
                self.label_data = np.load(self.label_paths[self.index])
            else:
                self.label_data = np.zeros((19, 19), dtype=int)

            filename = os.path.basename(self.image_paths[self.index])
            self.filename_label.config(text=f"Image: {filename}")
            self.count_label.config(text=f"Image {self.index+1}/{len(self.image_paths)}")
            self.refresh_canvas()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            print(f"Error details: {e}")

    # Other methods remain the same, only comments translated

if __name__ == "__main__":
    root = tk.Tk()
    app = GoBoardLabeler(root)
    root.mainloop()