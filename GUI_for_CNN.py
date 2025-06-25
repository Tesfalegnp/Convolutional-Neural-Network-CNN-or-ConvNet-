import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = load_model('tomato_model.keras')  # Make sure to have the model file in the same directory

# Class names
class_names = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Threshold for tomato leaf detection
TOMATO_THRESHOLD = 0.5

class TomatoDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tomato Leaf Disease Classifier")
        self.root.geometry("800x600")
        
        # Create GUI components
        self.create_widgets()
        
        # Initialize image path
        self.image_path = None
    
    def create_widgets(self):
        # Frame for image display
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(pady=20)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()
        
        # Button to select image
        self.select_btn = ttk.Button(
            self.root, 
            text="Select Tomato Leaf Image",
            command=self.load_image
        )
        self.select_btn.pack(pady=10)
        
        # Button to classify
        self.classify_btn = ttk.Button(
            self.root, 
            text="Classify Disease",
            command=self.classify_disease,
            state=tk.DISABLED
        )
        self.classify_btn.pack(pady=10)
        
        # Results display
        self.result_frame = ttk.LabelFrame(self.root, text="Classification Results")
        self.result_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        self.result_label = ttk.Label(
            self.result_frame, 
            text="Please select an image first",
            font=("Helvetica", 12)
        )
        self.result_label.pack(pady=20)
        
        # Confidence display
        self.confidence_label = ttk.Label(
            self.result_frame,
            text="",
            font=("Helvetica", 10)
        )
        self.confidence_label.pack(pady=10)
        
        # Detailed results
        self.details_label = ttk.Label(
            self.result_frame,
            text="",
            font=("Helvetica", 10),
            wraplength=700
        )
        self.details_label.pack(pady=10, padx=20)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return
        
        self.image_path = file_path
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk
        self.classify_btn.config(state=tk.NORMAL)
        self.result_label.config(text="Image loaded. Click 'Classify Disease'")
        self.confidence_label.config(text="")
        self.details_label.config(text="")
    
    def classify_disease(self):
        if not self.image_path:
            return
        
        # Preprocess the image
        img = image.load_img(self.image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
        max_prob = np.max(predictions)
        predicted_class = np.argmax(predictions)
        
        # Check if it's a tomato leaf
        if max_prob < TOMATO_THRESHOLD:
            self.result_label.config(
                text="Not a Tomato Leaf!",
                foreground="red"
            )
            self.confidence_label.config(text="")
            self.details_label.config(text="Please select an image of a tomato leaf.")
            return
        
        # Get results
        disease_name = class_names[predicted_class]
        confidence = max_prob * 100
        
        # Display results
        self.result_label.config(
            text=f"Predicted: {disease_name}",
            foreground="green" if "healthy" in disease_name else "orange"
        )
        self.confidence_label.config(
            text=f"Confidence: {confidence:.2f}%"
        )
        
        # Show detailed probabilities
        details = "Probabilities:\n"
        for i, prob in enumerate(predictions[0]):
            details += f"{class_names[i]}: {prob*100:.2f}%\n"
        self.details_label.config(text=details)

if __name__ == "__main__":
    root = tk.Tk()
    app = TomatoDiseaseApp(root)
    root.mainloop()