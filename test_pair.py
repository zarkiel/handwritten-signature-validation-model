
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model
from functions import euclidean_distance, contrastive_loss
from preprocessing import preprocess_image_bytes_from_file

MODEL_PATH = "models/best_model.keras"
THRESHOLD = 0.1899

model = load_model(MODEL_PATH, custom_objects={
    "euclidean_distance": euclidean_distance,
    "contrastive_loss": contrastive_loss
})

image1 = preprocess_image_bytes_from_file("dataset/cedar1/full_org/original_1_1.png")
image2 = preprocess_image_bytes_from_file("dataset/cedar1/full_org/original_1_2.png")

distance = float(model.predict([image1, image2])[0][0])

result = distance <= THRESHOLD