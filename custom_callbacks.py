import keras

class LogEpochStats(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        logs=logs or {}
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))