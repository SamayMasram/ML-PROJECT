from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory
import tensorflow as tf
import numpy as np
import json, os, webbrowser, io, base64, traceback
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# === CONFIG ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = 'my_model.h5'  # default model filename to try to auto-load
MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_FILENAME)
UPLOAD_FOLDER = os.path.join(SCRIPT_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hardcoded mapping (used if class_indices.json not found)
HARDCODED_CLASS_INDICES = {
    'Anthracnose': 0, 'algal leaf': 1, 'bird eye spot': 2, 'brown blight': 3,
    'gray light': 4, 'healthy': 5, 'red leaf spot': 6, 'white spot': 7
}

# === Utility: safe model load ===
MODEL = None
MODEL_LOAD_MSG = 'No model loaded.'
try:
    if os.path.exists(MODEL_PATH):
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        MODEL_LOAD_MSG = f'Model loaded from {MODEL_PATH}'
    else:
        MODEL_LOAD_MSG = f'No model file at {MODEL_PATH}. Upload via web UI.'
except Exception as e:
    MODEL = None
    MODEL_LOAD_MSG = f'Failed to load model: {e}'
    print('Model load error:')
    traceback.print_exc()

# === Class mapping load ===
CLASS_FILE = os.path.join(SCRIPT_DIR, 'class_indices.json')
if os.path.exists(CLASS_FILE):
    try:
        with open(CLASS_FILE, 'r', encoding='utf-8') as f:
            nm2i = json.load(f)
        IDX2NAME = {int(v): k for k, v in nm2i.items()}
    except Exception:
        IDX2NAME = {v: k for k, v in HARDCODED_CLASS_INDICES.items()}
else:
    IDX2NAME = {v: k for k, v in HARDCODED_CLASS_INDICES.items()}

# === Helpers ===

def image_to_base64(img: Image.Image, fmt='PNG'):
    buff = io.BytesIO()
    img.save(buff, format=fmt)
    return 'data:image/{};base64,'.format(fmt.lower()) + base64.b64encode(buff.getvalue()).decode('utf-8')

def infer_target_size_from_model():
    """Try to determine expected (height,width) for image input.
    Returns (h,w) or None if unable.
    """
    if MODEL is None:
        return None
    try:
        ish = MODEL.input_shape
        if isinstance(ish, (list, tuple)):
            # If list, take first
            if isinstance(ish[0], (list, tuple)):
                ish = ish[0]
        if not ish:
            return None
        # Typical image shape: (None, h, w, c)
        if len(ish) == 4:
            return (int(ish[1]) if ish[1] else None, int(ish[2]) if ish[2] else None)
        # Some models accept (None, h, w) or (None, n)
        return None
    except Exception:
        return None


def preprocess_for_model(pil_img: Image.Image):
    """Preprocess an input PIL image to the shape the model expects.
    Attempts several reasonable fallbacks to avoid input-shape mismatches.
    Returns (x_np, message) where x_np is the numpy array ready for model.predict and message is a text log.
    """
    log = []
    # If model expects flat 2D input (None, N) try to produce that
    if MODEL is not None:
        try:
            inshape = MODEL.input_shape
            log.append(f'model.input_shape={inshape}')
            # Normalize to canonical tuple if list
            if isinstance(inshape, list):
                inshape = inshape[0]
            # If model expects flattened vector
            if len(inshape) == 2 and inshape[1] is not None:
                expected = int(inshape[1])
                log.append(f'model expects flat vector of length {expected}')
                # Attempt to find a reasonable image resize so that h*w*3 == expected
                if expected % 3 == 0:
                    pix = expected // 3
                    h = int(round(pix ** 0.5))
                    if h * h == pix:
                        log.append(f'resizing image to ({h},{h}) and flattening')
                        img = pil_img.convert('RGB').resize((h,h))
                        arr = np.array(img).astype(np.float32).reshape(-1) / 255.0
                        arr = arr.reshape((1, expected))
                        return arr, '\n'.join(log)
                    else:
                        # choose close factors
                        w = int(round(pix / h))
                        log.append(f'resizing image to approx ({h},{w}) and flattening')
                        img = pil_img.convert('RGB').resize((w,h))
                        arr = np.array(img).astype(np.float32).reshape(-1)
                        if arr.size >= expected:
                            arr = arr[:expected]
                        else:
                            # pad
                            arr = np.pad(arr, (0, expected - arr.size), mode='constant')
                        arr = arr.reshape((1, expected)) / 255.0
                        return arr, '\n'.join(log)
                # As last resort, flatten resized 224x224 and then either trim or pad
                fallback_size = 224
                img = pil_img.convert('RGB').resize((fallback_size, fallback_size))
                arr = np.array(img).astype(np.float32).reshape(-1)
                if arr.size >= expected:
                    arr = arr[:expected]
                else:
                    arr = np.pad(arr, (0, expected - arr.size), mode='constant')
                arr = arr.reshape((1, expected)) / 255.0
                log.append(f'fallback: resized to {fallback_size}x{fallback_size}, trimmed/padded to {expected}')
                return arr, '\n'.join(log)
            # If model expects image tensor (None,h,w,c)
            if len(inshape) == 4 and inshape[1] is not None and inshape[2] is not None:
                th, tw = int(inshape[1]), int(inshape[2])
                log.append(f'model expects image shape ({th},{tw})')
                img = pil_img.convert('RGB').resize((tw, th))
                arr = np.array(img).astype(np.float32) / 255.0
                arr = np.expand_dims(arr, 0)
                return arr, '\n'.join(log)
        except Exception as e:
            log.append(f'Error while inferring input shape: {e}')
    # If we couldn't infer, default to 224x224 RGB
    log.append('Could not infer shape; defaulting to (224,224) RGB')
    img = pil_img.convert('RGB').resize((224,224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr, '\n'.join(log)

# === HTML template (modern look) ===
HTML = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Tea Leaf Disease Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { background: linear-gradient(180deg, #f6f9fc 0%, #ffffff 100%); font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
      .card { border: none; border-radius: 16px; box-shadow: 0 8px 30px rgba(37, 52, 75, 0.08); }
      .preview { max-width: 100%; border-radius: 12px; }
      .prob-row { display:flex; justify-content:space-between; }
      .muted { color:#6b7280 }
      .small-card { background:#f8f9fb; padding:10px; border-radius:8px }
    </style>
  </head>
  <body>
    <div class="container py-5">
      <div class="row justify-content-center">
        <div class="col-md-8">
          <div class="card p-4">
            <h3 class="mb-3 text-center">Tea Leaf Disease Predictor 🌿</h3>
            <p class="text-center muted">Upload a leaf image and get the predicted disease. Model status: <strong>{{ model_status }}</strong></p>
            <form method="post" action="/predict" enctype="multipart/form-data" class="mb-3">
              <div class="input-group">
                <input type="file" name="image" accept="image/*" class="form-control" required>
                <button class="btn btn-primary" type="submit">Predict</button>
              </div>
            </form>

            <form method="post" action="/upload_model" enctype="multipart/form-data" class="mb-4">
              <div class="input-group">
                <input type="file" name="model_file" class="form-control">
                <button class="btn btn-outline-secondary" type="submit">Upload & Load Model (.h5)</button>
              </div>
            </form>

            {% if img_data %}
              <div class="row">
                <div class="col-md-6">
                  <img src="{{ img_data }}" class="preview" alt="uploaded image">
                </div>
                <div class="col-md-6">
                  <div class="small-card">
                    <h5>Prediction</h5>
                    <p><strong>{{ predicted_name }}</strong></p>
                    <p class="muted small">Index: {{ predicted_index }}</p>
                    {% if probs %}
                      <hr>
                      <h6>Top probabilities</h6>
                      {% for label, p in probs %}
                        <div class="prob-row"><div>{{ label }}</div><div>{{ '{:.2%}'.format(p) }}</div></div>
                      {% endfor %}
                    {% endif %}
                  </div>
                </div>
              </div>
            {% endif %}

            <hr>
            <details>
              <summary>Class mapping (index → name)</summary>
              <pre>{{ class_map }}</pre>
            </details>

            {% if log %}
              <hr>
              <details open>
                <summary>Preprocess log / debug info</summary>
                <pre>{{ log }}</pre>
              </details>
            {% endif %}

          </div>
        </div>
      </div>
    </div>
  </body>
</html>
'''

# === Routes ===
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML, model_status=MODEL_LOAD_MSG, img_data=None, predicted_name=None, predicted_index=None, probs=None, class_map=json.dumps({k:v for v,k in IDX2NAME.items()}, indent=2), log=None)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    global MODEL, MODEL_LOAD_MSG, MODEL_PATH
    mf = request.files.get('model_file')
    if not mf or mf.filename == '':
        return redirect(url_for('index'))
    filename = secure_filename(mf.filename)
    save_path = os.path.join(SCRIPT_DIR, filename)
    mf.save(save_path)
    try:
        MODEL = tf.keras.models.load_model(save_path)
        MODEL_PATH = save_path
        MODEL_LOAD_MSG = f'Model loaded from {save_path}'
    except Exception as e:
        MODEL = None
        MODEL_LOAD_MSG = f'Failed to load model: {e}'
        print('Model load error:')
        traceback.print_exc()
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file or file.filename == '':
        return redirect(url_for('index'))
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    pil_img = Image.open(save_path)
    x, log = preprocess_for_model(pil_img)

    if MODEL is None:
        return render_template_string(HTML, model_status=MODEL_LOAD_MSG, img_data=image_to_base64(pil_img), predicted_name='No model loaded', predicted_index=None, probs=None, class_map=json.dumps({k:v for v,k in IDX2NAME.items()}, indent=2), log=log)

    try:
        pred = MODEL.predict(x)
        # classification-style output handling
        probs = None
        predicted_index = None
        predicted_name = None
        if pred.ndim == 2 and pred.shape[1] > 1:
            probs_arr = pred[0]
            # get top 5
            idxs = probs_arr.argsort()[::-1]
            probs = []
            for i in idxs[:5]:
                name = IDX2NAME.get(int(i), f'index_{i}')
                probs.append((name, float(probs_arr[int(i)])))
            predicted_index = int(idxs[0])
            predicted_name = IDX2NAME.get(predicted_index, f'index_{predicted_index}')
        else:
            predicted_name = str(pred.tolist())

        return render_template_string(HTML, model_status=MODEL_LOAD_MSG, img_data=image_to_base64(pil_img), predicted_name=predicted_name, predicted_index=predicted_index, probs=probs, class_map=json.dumps({k:v for v,k in IDX2NAME.items()}, indent=2), log=log)
    except Exception as e:
        tb = traceback.format_exc()
        print('Prediction error:')
        print(tb)
        return render_template_string(HTML, model_status=MODEL_LOAD_MSG, img_data=image_to_base64(pil_img), predicted_name=f'Prediction failed: {e}', predicted_index=None, probs=None, class_map=json.dumps({k:v for v,k in IDX2NAME.items()}, indent=2), log=log + '\n\n' + tb)

if __name__ == '__main__':
    # auto-open browser once
    webbrowser.open('http://127.0.0.1:5000')
    # disable reloader to avoid double model loads and double browser opens
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
