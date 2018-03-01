import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import models, layers, optimizers
import glob
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
import time
import multiprocessing
import json
import logging
import tensorflow as tf
from keras.utils import multi_gpu_model
import configparser

config = configparser.ConfigParser()
basepath = os.path.dirname(__file__)
config.read(os.path.abspath(os.path.join(basepath, 'configurations.ini')))

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -10s %(funcName) -15s %(lineno) -5d: %(message)s')

check = config['LOGGING']['log_to_file']
if check.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
    logging.basicConfig(level=logging.INFO,
                    format=LOG_FORMAT, filename=os.path.join(config['LOGGING']['log_file_dir'],
                                                             config['LOGGING']['log_file_name']))
else:
    logging.basicConfig(level=logging.INFO,
                        format=LOG_FORMAT)


class LogEpochStats(Callback):
    def __init__(self):
        super(LogEpochStats, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        LOGGER.info('Epoch {} started.'.format(epoch+1))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        LOGGER.info('Batch {}, Accuracy -> {}, '
                    'Loss -> {}, '
                    .format(batch, logs.get('acc'), logs.get('loss')))


class CustomModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, custom_model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.custom_model = custom_model

        if mode not in ['auto', 'min', 'max']:
            # warnings.warn('ModelCheckpoint mode %s is unknown, '
            #               'fallback to auto mode.' % (mode),
            #               RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    pass
                    # warnings.warn('Can save best model only with %s available, '
                    #               'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.custom_model.save_weights(filepath, overwrite=True)
                        else:
                            self.custom_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.custom_model.save_weights(filepath, overwrite=True)
                else:
                    self.custom_model.save(filepath, overwrite=True)



class TransferLearning(object):

    def __init__(self, train_dir, val_dir):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.datagen = ImageDataGenerator(rescale=1./255)
        self.batch_size = int(config['MODELLING']['batch_size'])
        self.IM_WIDTH = int(config['MODELLING']['im_width'])
        self.IM_HEIGHT = int(config['MODELLING']['im_height'])
        self.epochs = int(config['MODELLING']['epochs'])
        self.model_driectory = config['MODELLING']['model_directory']
        self.tensorboard_logs_dir = config['MODELLING']['tensorboard_logs_dir']
        self.model_name = 'tl_inceptionv3.{epoch:02d}-{val_loss:.2f}.hdf5'
        self.tensorboard_logs_name = "tl_inceptionv3_{}".format(time.time())
        self.cpu_count = int(config['MODELLING']['cpu_count'])      # multiprocessing.cpu_count()
        self.FC_SIZE = int(config['MODELLING']['last_layer_fc_size'])
        # self.fraction_to_unfreeze_while_fine_tuning = 0.4
        self.num_layers_to_freeze_while_fine_tuning = int(config['MODELLING']['num_layers_to_freeze_while_fine_tuning'])
        self.checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_driectory,
                                                           self.model_name),
                                          monitor=config['MODEL_CHECKPOINT']['monitor'],
                                          save_best_only=True, save_weights_only=True)

        self.tensorboard = TensorBoard(
            log_dir=os.path.join(self.tensorboard_logs_dir, self.tensorboard_logs_name))
        self.custom_logger = LogEpochStats()
        self.num_gpus = int(config['MODELLING']['num_gpus'])


    def __unfreeze_layers_in_model(self):
        # conv_base.trainable = True

        # num_layers = len(conv_base.layers)
        # NB_IV3_LAYERS_TO_FREEZE = num_layers - int(self.fraction_to_unfreeze_while_fine_tuning * num_layers)
        LOGGER.info('Number of trainable weights before unfreezing the conv base in model: {}'.format(
            len(self.model.trainable_weights)))

        for layer in self.conv_base.layers[:self.num_layers_to_freeze_while_fine_tuning]:
            layer.trainable = False
        for layer in self.conv_base.layers[self.num_layers_to_freeze_while_fine_tuning:]:
            layer.trainable = True

        # LOGGER.info('Number of trainable weights after unfreezing the conv base: {}'.format(len(conv_base.trainable_weights)))
        LOGGER.info('Number of trainable weights after unfreezing the conv base in model: {}'.format(len(self.model.trainable_weights)))

        # return model


    def __freeze_conv_base(self, model, conv_base):
        LOGGER.info('Number of trainable weights before freezing the conv base: {}'.format(len(model.trainable_weights)))
        for layer in conv_base.layers:
            layer.trainable = False
        LOGGER.info('Number of trainable weights after freezing the conv base: {}'.format(len(model.trainable_weights)))
        return model, conv_base


    def __count_number_of_files(self, root_directory):
        count = 0
        for files in glob.glob(os.path.join(root_directory, '*')):
            count += len(os.listdir(files))
        return count


    def __training_essentials(self):
        self.NUM_CLASSES = len(os.listdir(self.train_dir))
        self.NUM_TRAINING_SAMPLES = self.__count_number_of_files(self.train_dir)
        self.NUM_VALIDATION_SAMPLES = self.__count_number_of_files(self.val_dir)
        if not os.path.isdir(self.model_driectory):
            os.makedirs(self.model_driectory)
        self.validation_steps = self.NUM_VALIDATION_SAMPLES // self.batch_size
        self.steps_per_epoch = self.NUM_TRAINING_SAMPLES // self.batch_size # calculate this based on your number of training samples,
        # batch size and amount of augmented sample you want to give (it can also be num_train_samples//batch-size)


    def __create_conv_base(self, transfer_learn=True):
        conv_base = InceptionV3(weights=None,
                                    include_top=False,
                                    input_shape=(self.IM_HEIGHT, self.IM_WIDTH, 3))
        # self.conv_base.summary()
        if transfer_learn:
            LOGGER.info('Loading InceptionV3 weights for transfer learning')
            conv_base.load_weights(config['MODELLING']['model_path'])
        return conv_base


    def __create_generators(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                                 target_size=(self.IM_WIDTH, self.IM_HEIGHT),
                                                                 batch_size=self.batch_size)

        self.validation_generator = test_datagen.flow_from_directory(self.val_dir,
                                                                     target_size=(self.IM_HEIGHT, self.IM_WIDTH),
                                                                     batch_size=self.batch_size)

        LOGGER.info('Saving train generator class indices...')
        with open('train_class_indices.json', 'w') as fp:
            json.dump(self.train_generator.class_indices, fp)

        LOGGER.info('Saving val generator class indices...')
        with open('val_class_indices.json', 'w') as fp:
            json.dump(self.validation_generator.class_indices, fp)



    def __train_model(self, fine_tuning=False):
        if fine_tuning:
            learning_rate = 0.00001
        else:
            learning_rate = 0.0001

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.RMSprop(lr=learning_rate),
                           metrics=['acc'])

        if fine_tuning:
            model_as_json = self.model.to_json()
            with open('self_model_after_fine_tuning_before_fit_generator.json', 'w') as f:
                f.write(model_as_json)
        else:
            model_as_json = self.model.to_json()
            with open('self_model_after_transfer_before_fit_generator.json', 'w') as f:
                f.write(model_as_json)

        history = self.model.fit_generator(self.train_generator,
                                       steps_per_epoch=self.steps_per_epoch,
                                       epochs=self.epochs,
                                       validation_data=self.validation_generator,
                                       validation_steps=self.validation_steps,
                                       callbacks=[self.tensorboard, self.custom_logger, CustomModelCheckpoint(self.model, self.model_name)],
                                       workers=self.cpu_count)

        # self.plot(history)
        if fine_tuning:
            self.model.save('self_model_after_fine_tuning_after_fit_generator.h5')
            self.model.save_weights(
                os.path.join(self.model_driectory, 'self_model_weights_after_fine_tuning_after_fit_generator.h5'))
        else:
            self.model.save('self_model_after_transfer_learning_after_fit_generator.h5')
            self.model.save_weights(
                os.path.join(self.model_driectory, 'self_model_weights_after_transfer_learning_after_fit_generator.h5'))
        # return model


    def __train_parallel_model(self, model, fine_tuning=False):
        if fine_tuning:
            learning_rate = 0.00001
        else:
            learning_rate = 0.0001

        model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.RMSprop(lr=learning_rate),
                           metrics=['acc'])

        if fine_tuning:
            # model.save('model_after_fine_tuning_before_fit_generator.h5')
            # model_as_json = model.to_json()
            # with open('fine_tune_model_before_fit_generator.json', 'w') as f:
            #     f.write(model_as_json)
            self.model.save('self_model_after_fine_tuning_before_fit_generator.h5')
            model_as_json = self.model.to_json()
            with open('self_model_after_fine_tuning_before_fit_generator.json', 'w') as f:
                f.write(model_as_json)
        else:
            # model.save('model_after_transfer_learning_before_fit_generator.h5')
            # model_as_json = model.to_json()
            # with open('transfer_learning_model_before_fit_generator.json', 'w') as f:
            #     f.write(model_as_json)
            self.model.save('self_model_after_transfer_learning_before_fit_generator.h5')
            model_as_json = self.model.to_json()
            with open('self_model_after_transfer_before_fit_generator.json', 'w') as f:
                f.write(model_as_json)

        history = model.fit_generator(self.train_generator,
                                       steps_per_epoch=self.steps_per_epoch,
                                       epochs=self.epochs,
                                       validation_data=self.validation_generator,
                                       validation_steps=self.validation_steps,
                                       callbacks=[self.checkpoint, self.tensorboard, self.custom_logger, CustomModelCheckpoint(self.model)],
                                       workers=self.cpu_count)

        # self.plot(history)
        if fine_tuning:
            # model.save('model_after_fine_tuning_after_fit_generator.h5')
            # model_as_json = model.to_json()
            # with open('model_after_fine_tuning_after_fit_generator.json', 'w') as f:
            #     f.write(model_as_json)
            # model.save_weights(os.path.join(self.model_driectory ,'model_weights_after_fine_tuning_after_fit_generator.h5'))

            self.model.save('self_model_after_fine_tuning_after_fit_generator.h5')
            model_as_json = self.model.to_json()
            with open('self_model_after_fine_tuning_after_fit_generator.json', 'w') as f:
                f.write(model_as_json)
            self.model.save_weights(
                os.path.join(self.model_driectory, 'self_model_weights_after_fine_tuning_after_fit_generator.h5'))
        else:
            # model.save('model_after_transfer_learning_after_fit_generator.h5')
            # model_as_json = model.to_json()
            # with open('model_after_transfer_after_fit_generator.json', 'w') as f:
            #     f.write(model_as_json)
            # model.save_weights(os.path.join(self.model_driectory, 'model_weights_after_transfer_learning_after_fit_generator.h5'))

            self.model.save('self_model_after_transfer_learning_after_fit_generator.h5')
            model_as_json = self.model.to_json()
            with open('self_model_after_transfer_after_fit_generator.json', 'w') as f:
                f.write(model_as_json)
            self.model.save_weights(
                os.path.join(self.model_driectory, 'self_model_weights_after_transfer_learning_after_fit_generator.h5'))
        # return model


    def __create_model(self, conv_base):
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(self.FC_SIZE, activation='relu'))
        model.add(layers.Dense(self.NUM_CLASSES, activation='softmax'))
        model.summary()
        check_load_weights = config['MODELLING']['load_weights_of_previous_model']
        if check_load_weights.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
            LOGGER.info('Loading weights of previous model...')
            model.load_weights(config['MODELLING']['weights_of_previous_model'])
        return model


    def __plot(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc)+1)

        plt.plot(epochs, acc, 'g', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()

        plt.plot(epochs, loss, 'g', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()


    def __build_model(self, transfer_learn=True):
        # create conv base
        LOGGER.info('Creating Inception V3 conv base')
        conv_base = self.__create_conv_base(transfer_learn=transfer_learn)
        # create model with conv_base
        model = self.__create_model(conv_base)
        # freeze conv base layers
        model, conv_base = self.__freeze_conv_base(model, conv_base)
        return model, conv_base


    def begin_training(self):
        # calculate training essentials
        LOGGER.info('Calulating num_classes, samples, creating model directory, etc...')
        self.__training_essentials()
        self.__create_generators()
        check_tl = config['MODELLING']['train_from_scratch']
        if check_tl.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
            check_tl = False
        else:
            check_tl = True
        if self.num_gpus <= 1:
            self.model, self.conv_base = self.__build_model(transfer_learn=check_tl)
            # transfer learn with frozen conv base
            self.__train_model()
            if check_tl:
                # unfreeze conv base layers
                self.__unfreeze_layers_in_model()
                # # fine tune with unfrozen layers
                self.__train_model(fine_tuning=True)
        else:
            with tf.device("/cpu:0"):
                self.model, self.conv_base = self.__build_model(transfer_learn=check_tl)
            parallel_model = multi_gpu_model(self.model, gpus=self.num_gpus)
            # transfer learn with frozen conv base
            parallel_model = self.__train_parallel_model(parallel_model)
            if check_tl:
                # unfreeze conv base layers
                parallel_model = self.__unfreeze_layers_in_model(parallel_model, self.conv_base)
                # # fine tune with unfrozen layers
                parallel_model = self.__train_model(parallel_model, fine_tuning=True)



if __name__ == '__main__':
    try:
        tl_instance = TransferLearning(train_dir=config['FILEPATHS']['train_dir'],
                                       val_dir=config['FILEPATHS']['val_dir'])
        LOGGER.info('Beginning Training Process')
        tl_instance.begin_training()
    except Exception as e:
        LOGGER.error(e, exc_info=True)
