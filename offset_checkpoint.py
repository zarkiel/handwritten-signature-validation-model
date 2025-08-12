from tensorflow.keras.callbacks import ModelCheckpoint

class OffsetCheckpoint(ModelCheckpoint):
    def __init__(self, offset=0, **kwargs):
        self.offset = offset
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        # Guardar con el número de época real
        adjusted_epoch = epoch + self.offset
        self.filepath = f"models/model_epoch_{adjusted_epoch:02d}.keras"
        super().on_epoch_end(epoch, logs)