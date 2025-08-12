import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Lambda)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from offset_checkpoint import OffsetCheckpoint
from signature_data_generator import SignatureDataGenerator
from functions import euclidean_distance, contrastive_loss
from model import create_base_network

np.random.seed(1337)
tf.random.set_seed(1337)

img_height, img_width = 155, 220
input_shape = (img_height, img_width, 1)
batch_size = 32
epochs = 10

dataset = "cedar1"
pairs_file = f'dataset/{dataset}/signature_pairs.csv'
model_save_path = 'models/best_model.keras'
current_epoch = 1

if os.path.exists(model_save_path):
    print(f"Loading existing model from {model_save_path}")
    model = load_model(model_save_path, custom_objects={'contrastive_loss': contrastive_loss})
else:
    print("Creating a new Siamese model...")
    base_network = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance)([processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    model.compile(loss=contrastive_loss, optimizer=RMSprop(learning_rate=1e-4))

# Callbacks
checkpoint_all = OffsetCheckpoint(
    offset=current_epoch,
    filepath='models/model_epoch_{epoch:02d}.keras',
    save_freq='epoch',
    verbose=1
)

checkpoint_best = ModelCheckpoint(
    filepath=model_save_path,
    monitor='val_loss', save_best_only=True, verbose=1
)

gen = SignatureDataGenerator(dataset=dataset, pairs_file=pairs_file)


output_signature = (
    (
        tf.TensorSpec(shape=(None, img_height, img_width, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, img_height, img_width, 1), dtype=tf.float32)
    ),
    tf.TensorSpec(shape=(None,), dtype=tf.float32) 
) 

train_dataset = tf.data.Dataset.from_generator(
    lambda: gen.train_generator(batch_size),
    output_signature=output_signature
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: gen.validation_generator(batch_size),
    output_signature=output_signature
)

model.fit(
    train_dataset,
    steps_per_epoch=960,
    validation_data=val_dataset,
    validation_steps=120,
    epochs=epochs,
    callbacks=[checkpoint_all, checkpoint_best]
)
