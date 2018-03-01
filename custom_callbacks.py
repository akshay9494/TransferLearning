from keras.callbacks import Callback

class LogEpochStats(Callback):
    def __init__(self, steps_per_epoch, logger):
        super(LogEpochStats, self).__init__()
        self.steps_per_epoch = steps_per_epoch
        self.logger = logger

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        self.logger.info('Epoch {} started.'.format(epoch + 1))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.logger.info('Batch {}/{}, Accuracy -> {}, '
                    'Loss -> {}'
                    .format(batch, self.steps_per_epoch, logs.get('acc'), logs.get('loss')))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.logger.info('Epoch {} ended.'.format(epoch + 1))
        self.logger.info('Train accuracy -> {}, Train Loss -> {}, Validation Accuracy -> {}, Validation loss -> {}'
                    .format(logs.get('acc'), logs.get('loss'), logs.get('val_acc'), logs.get('val_loss')))