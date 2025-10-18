import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# --- Configuration ---
# Update these variables for your specific model
MODEL_PATH = 'cifar10_model.keras'
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- Main Application Class ---
class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry("500x550")
        self.root.configure(bg="#f0f0f0")

        # Load the trained model
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load model: {e}")
            self.root.destroy()
            return

        # --- Create GUI Widgets ---
        self.title_label = tk.Label(root, text="Image Predictor", font=("Helvetica", 20, "bold"), bg="#f0f0f0")
        self.title_label.pack(pady=10)

        self.image_label = tk.Label(root, text="No Image Selected", bg="#cccccc", width=40, height=20)
        self.image_label.pack(pady=10)

        self.prediction_label = tk.Label(root, text="Prediction: -", font=("Helvetica", 14), bg="#f0f0f0")
        self.prediction_label.pack(pady=20)

        self.select_button = tk.Button(root, text="Select Image", font=("Helvetica", 12), command=self.select_image)
        self.select_button.pack(pady=10)

    def select_image(self):
        """Opens a file dialog to select an image and predicts its class."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if not file_path:
            return # User cancelled the dialog

        try:
            # Open and display the image
            img = Image.open(file_path)
            img.thumbnail((300, 300)) # Resize for display
            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo, text="")
            self.image_label.image = photo

            # Predict the image class
            self.predict(file_path)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open or process image: {e}")

    def predict(self, file_path):
        """Preprocesses the image and uses the model to predict its class."""
        try:
            # 1. Load image and resize to model's expected input size
            img = Image.open(file_path).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            
            # 2. Convert image to a NumPy array
            img_array = np.array(img)
            
            # 3. Normalize the image (if your model expects it)
            img_array = img_array / 255.0
            
            # 4. Add a batch dimension
            # Model expects (batch_size, height, width, channels), e.g., (1, 32, 32, 3)
            img_array = np.expand_dims(img_array, axis=0)

            # 5. Make a prediction
            predictions = self.model.predict(img_array)
            score = tf.nn.softmax(predictions[0]) # Get probabilities

            # 6. Get the class with the highest probability
            predicted_class_index = np.argmax(score)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = 100 * np.max(score)

            # 7. Update the prediction label
            result_text = f"Prediction: {predicted_class_name} ({confidence:.2f}%)"
            self.prediction_label.config(text=result_text)

        except Exception as e:
            self.prediction_label.config(text="Prediction failed!")
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")

# --- Run the application ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()# gui_predict.py
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
import os
import numpy as np
from PIL import Image, ImageTk
import traceback

# Try to import tensorflow, give friendly message on failure
try:
    import tensorflow as tf
except Exception as e:
    tk.Tk().withdraw()
    messagebox.showerror("TensorFlow error", f"Failed to import TensorFlow:\n{e}\n\nInstall with: pip install tensorflow")
    raise

MODEL = None
MODEL_PATH = None

def load_model_dialog():
    global MODEL, MODEL_PATH
    # Ask for file or directory
    tk.Tk().withdraw()
    sel = filedialog.askopenfilename(title="Select model (.h5) or cancel to choose a SavedModel folder",
                                      filetypes=[("Keras HDF5", "*.h5"), ("All files", "*.*")])
    if sel:
        # user picked a file (maybe .h5)
        MODEL_PATH = sel
    else:
        # ask for folder (SavedModel)
        folder = filedialog.askdirectory(title="Select SavedModel folder (or cancel to abort)")
        if folder:
            MODEL_PATH = folder
        else:
            messagebox.showinfo("Model load", "No model selected. Exiting.")
            return False

    try:
        # If path is a .h5 file -> load directly
        if os.path.isfile(MODEL_PATH) and MODEL_PATH.lower().endswith(".h5"):
            MODEL = tf.keras.models.load_model(MODEL_PATH)
        else:
            # assume SavedModel directory
            # TensorFlow will raise if it's not a model directory
            MODEL = tf.keras.models.load_model(MODEL_PATH)
        messagebox.showinfo("Model loaded", f"Model loaded from:\n{MODEL_PATH}")
        print("Model summary:")
        try:
            MODEL.summary()
        except Exception:
            pass
        return True
    except Exception as e:
        tb = traceback.format_exc()
        messagebox.showerror("Failed to load model", f"Error loading model:\n{e}\n\nSee console for full traceback.")
        print("Full traceback:\n", tb)
        return False

def preprocess_image_for_model(path, target_size=(224,224)):
    img = Image.open(path).convert("RGB").resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

def choose_and_predict():
    if MODEL is None:
        ok = load_model_dialog()
        if not ok:
            return
    p = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png;*.bmp;*.tif;*.tiff")])
    if not p:
        return
    try:
        # Try to infer input size from model, else fallback to 224
        input_shape = None
        try:
            # TensorFlow Keras: model.input_shape may be (None,h,w,c) or a list
            ish = MODEL.input_shape
            if isinstance(ish, (list, tuple)):
                # promote first entry
                if isinstance(ish[0], tuple):
                    ish = ish[0]
            if ish and len(ish) >= 3:
                # last two dims are h,w if channels_last; handle (None,h,w,c) or (None,c)
                if len(ish) == 4:
                    input_shape = (ish[1], ish[2])
                elif len(ish) == 3:
                    input_shape = (ish[1], ish[2])
        except Exception:
            input_shape = None

        if not input_shape:
            input_shape = (224,224)

        x = preprocess_image_for_model(p, target_size=input_shape)
        pred = MODEL.predict(x)
        # handle classification-like outputs
        if pred.ndim == 2 and pred.shape[1] > 1:
            cls = int(np.argmax(pred, axis=1)[0])
            score = pred[0].tolist()
            txt = f"Predicted class index: {cls}\nScores: {score}"
        else:
            # regression or single-output
            txt = f"Model output: {pred.tolist()}"
        # show image and result in GUI
        img = Image.open(p).resize((300,300))
        tkimg = ImageTk.PhotoImage(img)
        img_label.config(image=tkimg)
        img_label.image = tkimg
        result_label.config(text=txt)
        print("Prediction:", txt)
    except Exception as e:
        tb = traceback.format_exc()
        messagebox.showerror("Prediction error", f"Error while predicting:\n{e}\n\nSee console for full traceback.")
        print("Full traceback:\n", tb)

# Build simple GUI
root = tk.Tk()
root.title("Model Predictor")

btn_model = Button(root, text="Load model (.h5 or SavedModel)", command=load_model_dialog)
btn_model.pack(pady=6)

btn = Button(root, text="Open Image & Predict", command=choose_and_predict)
btn.pack(pady=6)

img_label = Label(root)
img_label.pack()

result_label = Label(root, text="No image loaded", justify="left")
result_label.pack(pady=8)

# Optionally auto-load a model if a file called my_model.h5 exists next to the script
script_dir = os.path.dirname(os.path.abspath(__file__))
auto_h5 = os.path.join(script_dir, "my_model.h5")
auto_saved = os.path.join(script_dir, "saved_model")
if os.path.exists(auto_h5) or os.path.exists(auto_saved):
    try:
        MODEL = tf.keras.models.load_model(auto_h5 if os.path.exists(auto_h5) else auto_saved)
        MODEL_PATH = auto_h5 if os.path.exists(auto_h5) else auto_saved
        print("Auto-loaded model from", MODEL_PATH)
    except Exception:
        pass

root.mainloop()
