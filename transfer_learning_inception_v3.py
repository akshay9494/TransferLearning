import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import models, layers, optimizers
import glob
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard, RemoteMonitor
import time
import multiprocessing
import json
import logging
import tensorflow as tf
from keras.utils import multi_gpu_model
import configparser


# TODO:-
# 1. update all hard coded values to be read from configuration files
# 2. Add multi_gpu_model from keras, to make it ready for production
# 3. Add documentation

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
        self.num_gpus = int(config['MODELLING']['num_gpus'])


    def __unfreeze_layers_in_model(self, model, conv_base):
        # conv_base.trainable = True

        # num_layers = len(conv_base.layers)
        # NB_IV3_LAYERS_TO_FREEZE = num_layers - int(self.fraction_to_unfreeze_while_fine_tuning * num_layers)
        LOGGER.info('Number of trainable weights before unfreezing the conv base in model: {}'.format(
            len(model.trainable_weights)))

        for layer in conv_base.layers[:self.num_layers_to_freeze_while_fine_tuning]:
            layer.trainable = False
        for layer in conv_base.layers[self.num_layers_to_freeze_while_fine_tuning:]:
            layer.trainable = True

        # LOGGER.info('Number of trainable weights after unfreezing the conv base: {}'.format(len(conv_base.trainable_weights)))
        LOGGER.info('Number of trainable weights after unfreezing the conv base in model: {}'.format(len(model.trainable_weights)))

        return model


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



    def __train_model(self, model, fine_tuning=False):
        if fine_tuning:
            learning_rate = 0.00001
        else:
            learning_rate = 0.0001

        model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.RMSprop(lr=learning_rate),
                           metrics=['acc'])

        history = model.fit_generator(self.train_generator,
                                           steps_per_epoch=self.steps_per_epoch,
                                           epochs=self.epochs,
                                           validation_data=self.validation_generator,
                                           validation_steps=self.validation_steps,
                                           callbacks=[self.checkpoint, self.tensorboard],
                                           workers=self.cpu_count)

        # self.plot(history)
        if fine_tuning:
            # model.save('model_after_fine_tuning.h5')
            model.save_weights('model_weights_after_fine_tuning.h5')
        else:
            # model.save('model_after_transfer_learning.h5')
            model.save_weights('model_weights_after_transfer_learning.h5')
        return model



    def __create_model(self, conv_base):
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(self.FC_SIZE, activation='relu'))
        model.add(layers.Dense(self.NUM_CLASSES, activation='softmax'))
        model.summary()
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
            model, conv_base = self.__build_model(transfer_learn=check_tl)
            # transfer learn with frozen conv base
            model = self.__train_model(model)
            if check_tl:
                # unfreeze conv base layers
                model = self.__unfreeze_layers_in_model(model, conv_base)
                # # fine tune with unfrozen layers
                model = self.__train_model(model, fine_tuning=True)
        else:
            with tf.device("/cpu:0"):
                model, conv_base = self.__build_model(transfer_learn=check_tl)
            parallel_model = multi_gpu_model(model, gpus=self.num_gpus)
            # transfer learn with frozen conv base
            parallel_model = self.__train_model(parallel_model)
            if check_tl:
                # unfreeze conv base layers
                parallel_model = self.__unfreeze_layers_in_model(parallel_model, conv_base)
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
