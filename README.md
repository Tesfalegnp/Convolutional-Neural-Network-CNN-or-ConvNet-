
# 🍅 Tomato Leaf Disease Detection using CNN

This project builds a deep learning-based image classification model to detect and classify diseases in tomato leaves. It uses a Convolutional Neural Network (CNN) trained on the [TomatoVillage dataset](#dataset) and includes pre-processing, training, evaluation, and inference components.

---

## 📌 Features


Let me know if you’d like a `LICENSE`, `requirements.txt`, or the Python script (`train.py`) version too!

- 📂 Train on images from a zipped dataset
- 🧠 Deep CNN model using TensorFlow/Keras
- 📉 Regularization with data augmentation & dropout to prevent overfitting
- 🎯 Accurate multi-class prediction of tomato leaf diseases
- ✅ Built-in detector: *Is this image a tomato leaf or not?*
- 💾 Save and reuse trained model (`.h5`)
- 📊 Includes accuracy and loss visualization
- 🖼️ Easy-to-use prediction function

---

## 🔧 Tech Stack

| Tool        | Purpose                            |
|-------------|------------------------------------|
| Python      | Main programming language          |
| TensorFlow  | Model training and evaluation      |
| Keras       | CNN layers and preprocessing       |
| Matplotlib  | Accuracy/loss visualization        |
| NumPy       | Numerical operations               |

---

## 📁 Dataset

- **Name:** TomatoVillage
- **Format:** `.zip`
- **Location:** `/content/drive/MyDrive/TomatoDataSets/TomatoVillage.zip`
- **Structure:**
```
TomatoVillage/
Tomato\_\_\_Healthy/
Tomato\_\_\_Late\_blight/
Tomato\_\_\_Leaf\_Mold/

```


> **Note:** Each class folder must begin with `"Tomato"` to trigger "tomato leaf check" logic.

---

## 🚀 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/Tesfalegnp/Convolutional-Neural-Network-CNN-or-ConvNet.git
cd Convolutional-Neural-Network-CNN-or-ConvNet
````

### 2. Upload Dataset

Place your zipped dataset at the following location:

```python
zip_path = "/content/drive/MyDrive/TomatoDataSets/TomatoVillage.zip"
```

### 3. Train the Model

Run the `train_model.ipynb` notebook or execute the `train.py` script:

```bash
python GUI.py
```

> Training automatically extracts and processes the dataset, augments images, and trains a CNN model.

---

## 🧪 Inference & Prediction

### Use `predict_image(img_path, model, class_names)` to make predictions.

```python
result = predict_image("/path/to/image.jpg", model, class_names)
print(result)
```

* If the image is not of a tomato leaf:
  🛑 `"Sorry, the image is not a tomato leaf."`
* If valid:
  ✅ `"Tomato leaf detected. Disease class: Tomato___Late_blight (95.23%)"`

---

## 📈 Training Results

After training, accuracy and loss graphs are displayed:

| Metric       | Result (Sample)                         |
| ------------ | --------------------------------------- |
| Training Acc | 98%                                     |
| Val Acc      | 96%                                     |
| Overfitting  | Mitigated with dropout and augmentation |

---

## 💾 Model Export

After training, the model is saved to:

```
/content/tomato_model.keras
```

Use this model in other environments like Flask, Tkinter, or Android apps.

---

## ✅ To Do

* [ ] Add Flask/Tkinter UI
* [ ] Convert to TensorFlow Lite for mobile apps
* [ ] Deploy on Streamlit or Hugging Face Spaces

---

## 🤝 Contributing

PRs and issues welcome! Please create a pull request or submit an issue.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* [TensorFlow](https://www.tensorflow.org/)
* [TomatoVillage Dataset](https://data.mendeley.com/)
* Inspiration from Potato Leaf CNN projects

---

## 📬 Contact

For questions, contact \[[Email-Tesfalegn](mailto:peterhope935@gmail.com)] or open an issue.

```
