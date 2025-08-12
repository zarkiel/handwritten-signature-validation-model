from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable

# Euclidean Distance and Contrastive Loss
@register_keras_serializable()
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

@register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def crop_from_xywhn_pil(image_pil, xywhn_box, padding=0):
    """Crop the image at the chosen box"""
    # Extract the coordinate from the box
    W, H = image_pil.size
    x_center, y_center, w, h = map(float, xywhn_box)

    x_c, y_c = x_center * W, y_center * H
    bw, bh = w * W, h * H
    
    x1 = max(int(x_c - bw / 2) - padding, 0)
    y1 = max(int(y_c - bh / 2) - padding, 0)
    x2 = min(int(x_c + bw / 2) + padding, W)
    y2 = min(int(y_c + bh / 2) + padding, H)

    # Crop out the image
    cropped = image_pil.crop((x1, y1, x2, y2))
    return cropped