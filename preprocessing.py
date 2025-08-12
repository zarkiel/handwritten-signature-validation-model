import numpy as np
import cv2
import base64
from PIL import Image
import io

img_height, img_width = 155, 220

def preprocess_image_bytes(file_storage):
    file_bytes = np.frombuffer(file_storage, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image(img)
    return np.expand_dims(img, axis=0)

def preprocess_image_base64(base64_string):
    file_bytes = base64.b64decode(base64_string)
    file_bytes = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image(img)
    return np.expand_dims(img, axis=0)

def preprocess_image_bytes_from_file(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image(img)
    return np.expand_dims(img, axis=0)

def preprocess_image(img):
    img = cv2.resize(img, (img_width, img_height))
    img = 255 - img  # invertir colores
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=-1)

def standardize(image):
    image -= np.mean(image)
    std = np.std(image)
    if std > 0:
        image /= std
    return image

def image_from_base64(base64_string):
    img_bytes = base64.b64decode(base64_string)
    img_buffer = io.BytesIO(img_bytes)
    document_image = Image.open(img_buffer)
    return document_image

def image_to_base64(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    base64_string = base64.b64encode(img_bytes).decode('utf-8')
    return base64_string