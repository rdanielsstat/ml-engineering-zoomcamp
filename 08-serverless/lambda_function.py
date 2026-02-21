import onnxruntime as ort
import numpy as np
from PIL import Image
from urllib import request
from io import BytesIO

# Download image
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

# Resize + RGB fix
def prepare_image(img, target_size = (200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img.resize(target_size, Image.NEAREST)

# Load the model
onnx_model_path = "hair_classifier_empty.onnx"
session = ort.InferenceSession(onnx_model_path, providers = ["CPUExecutionProvider"])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name

def lambda_handler(event, context = None):

    url = event["url"]

    # Download & resize
    img = download_image(url)
    img = prepare_image(img, (200, 200))

    # ---- full preprocessing ----
    arr = np.array(img).astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype = np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype = np.float32)
    arr = (arr - mean) / std

    # HWC → CHW → batch dimension
    arr = np.transpose(arr, (2, 0, 1))     # CHW
    inp = np.expand_dims(arr, axis=0)      # NCHW

    # run model
    # ---- Run inference ----
    result = session.run(None, {input_name: inp})[0]

    # simple output (depending on model)
    return float(result[0][0])