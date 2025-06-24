import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox

# === Load models safely ===
MODEL_PATHS = {
    "binary": "tomato_detector_model.keras",  # Binary classifier (tomato or not)
    "detailed": "tomato_model.keras"          # 10-class classifier
}

try:
    print("🔄 Loading models...")
    binary_model = load_model(MODEL_PATHS["binary"])
    detailed_model = load_model(MODEL_PATHS["detailed"])
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    messagebox.showerror("Model Error", f"Failed to load models:\n{e}")
    exit()

# === Define class labels ===
binary_class_names = ["Not Tomato", "Tomato"]

detailed_class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# === Image preprocessing ===
def preprocess_image(img_path, target_size=(256, 256)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        messagebox.showerror("Image Error", f"Image loading failed:\n{e}")
        return None

# === Prediction function ===
def predict_image(img_path):
    print(f"📸 Selected image: {img_path}")
    processed = preprocess_image(img_path)
    if processed is None:
        return "Image could not be processed."

    try:
        # First check if it's a tomato leaf
        is_tomato_pred = binary_model.predict(processed)[0]
        is_tomato = np.argmax(is_tomato_pred)
        tomato_confidence = is_tomato_pred[is_tomato]
        
        if not is_tomato:  # If not a tomato leaf
            return f"❌ Not a Tomato Leaf\nConfidence: {tomato_confidence:.2f}"
        
        # If it is a tomato leaf, do detailed classification
        detailed_preds = detailed_model.predict(processed)[0]
        pred_index = np.argmax(detailed_preds)
        confidence = detailed_preds[pred_index]
        condition = detailed_class_names[pred_index].split('___')[-1]
        
        return f"🍅 Tomato Leaf Detected\n🩺 Condition: {condition}\n📊 Confidence: {confidence:.2f}"
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        messagebox.showerror("Prediction Error", f"Prediction failed:\n{e}")
        return "Prediction failed."

# === GUI logic ===
def browse_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    try:
        result = predict_image(file_path)
        result_label.config(text=result)

        # Show image
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk
    except Exception as e:
        print(f"❌ GUI update error: {e}")
        messagebox.showerror("Display Error", f"Unable to show image or result:\n{e}")

# === GUI setup ===
root = tk.Tk()
root.title("🍅 Tomato Health Classifier")
root.geometry("400x550")
root.resizable(False, False)

title_label = Label(root, text="Tomato Health Classifier", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(root, text="📷 Select an image to begin", wraplength=350, font=("Helvetica", 12), justify="center")
result_label.pack(pady=20)

browse_button = Button(root, text="🔍 Browse Image", command=browse_and_predict, font=("Helvetica", 12), width=20)
browse_button.pack(pady=10)

root.mainloop()