import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import models, layers, optimizers
import glob
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from custom_callbacks import LogEpochStats, CustomModelCheckpoint
import json
import logging
import tensorflow as tf
from keras.utils import multi_gpu_model
import configparser
import time
from keras.models import load_model
import sys

config = configparser.ConfigParser()
basepath = os.path.dirname(__file__)
config.read(os.path.abspath(os.path.join(basepath, 'configurations.ini')))

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -10s %(funcName) -15s %(lineno) -5d: %(message)s')

check = config['LOGGING']['log_to_file']
if check.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
    if not os.path.isdir(config['LOGGING']['log_file_dir']):
        os.makedirs(config['LOGGING']['log_file_dir'])
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
        self.datagen = ImageDataGenerator(rescale=1. / 255)
        self.batch_size = int(config['MODELLING']['batch_size'])
        self.IM_WIDTH = int(config['MODELLING']['im_width'])
        self.IM_HEIGHT = int(config['MODELLING']['im_height'])
        self.tl_epochs = int(config['MODELLING']['tl_epochs'])
        self.ft_epochs = int(config['MODELLING']['ft_epochs'])
        self.model_directory = config['MODELLING']['model_directory']
        self.tensorboard_logs_dir = config['MODELLING']['tensorboard_logs_dir']
        self.tl_model_name = 'transfer_learning_inceptionv3_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'
        self.ft_model_name = 'fine_tuning_inceptionv3_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'
        self.tensorboard_logs_name = "tl_inceptionv3_{}".format(time.time())
        self.cpu_count = int(config['MODELLING']['cpu_count'])  # multiprocessing.cpu_count()
        self.FC_SIZE = int(config['MODELLING']['last_layer_fc_size'])
        self.num_layers_to_freeze_while_fine_tuning = int(config['MODELLING']['num_layers_to_freeze_while_fine_tuning'])
        self.tensorboard = TensorBoard(
            log_dir=os.path.join(self.tensorboard_logs_dir, self.tensorboard_logs_name))
        self.num_gpus = int(config['MODELLING']['num_gpus'])
        self.tl_lr = float(config['MODELLING']['transfer_learning_learning_rate'])
        self.ft_lr = float(config['MODELLING']['fine_tuning_learning_rate'])
        self.training_essentials_folder = config['MODELLING']['training_essentials_folder']


    def __count_number_of_files(self, root_directory):
        count = 0
        for files in glob.glob(os.path.join(root_directory, '*')):
            count += len(os.listdir(files))
        return count


    def __training_essentials(self):
        self.NUM_CLASSES = len(os.listdir(self.train_dir))
        LOGGER.info('Number of classes -> {}'.format(self.NUM_CLASSES))

        self.NUM_TRAINING_SAMPLES = self.__count_number_of_files(self.train_dir)
        LOGGER.info('Number of training samples -> {}'.format(self.NUM_TRAINING_SAMPLES))

        self.NUM_VALIDATION_SAMPLES = self.__count_number_of_files(self.val_dir)
        LOGGER.info('Number of Validation samples -> {}'.format(self.NUM_VALIDATION_SAMPLES))

        if not os.path.isdir(self.model_directory):
            os.makedirs(self.model_directory)
        if not os.path.isdir(self.tensorboard_logs_dir):
            os.makedirs(self.tensorboard_logs_dir)
        if not os.path.isdir(self.training_essentials_folder):
            os.makedirs(self.training_essentials_folder)

        self.validation_steps = self.NUM_VALIDATION_SAMPLES // self.batch_size
        self.steps_per_epoch = self.NUM_TRAINING_SAMPLES // self.batch_size  # calculate this based on your number of training samples,
        # batch size and amount of augmented sample you want to give (it can also be num_train_samples//batch-size)
        LOGGER.info('Steps per epoch -> {}'.format(self.steps_per_epoch))

        LOGGER.info('Setting up custom logger to log epoch stats.')
        self.custom_logger = LogEpochStats(steps_per_epoch=self.steps_per_epoch,
                                           logger=LOGGER)



    def __unfreeze_layers_in_model(self):
        LOGGER.info('Number of trainable weights before unfreezing the conv base in model: {}'.format(
            len(self.model.trainable_weights)))

        for layer in self.model.layers[0].layers[:self.num_layers_to_freeze_while_fine_tuning]:
            layer.trainable = False
        for layer in self.model.layers[0].layers[self.num_layers_to_freeze_while_fine_tuning:]:
            layer.trainable = True

        LOGGER.info('Number of trainable weights after unfreezing the conv base in model: {}'.format(
            len(self.model.trainable_weights)))


    def __freeze_conv_base(self, model):
        LOGGER.info(
            'Number of trainable weights before freezing the conv base: {}'.format(len(model.trainable_weights)))

        for layer in model.layers[0].layers:
            layer.trainable = False
        LOGGER.info('Number of trainable weights after freezing the conv base: {}'.format(len(model.trainable_weights)))
        return model


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

        LOGGER.info('Creating Training Generator.')
        self.train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                                 target_size=(self.IM_WIDTH, self.IM_HEIGHT),
                                                                 batch_size=self.batch_size)

        LOGGER.info('Creating validation Generator.')
        self.validation_generator = test_datagen.flow_from_directory(self.val_dir,
                                                                     target_size=(self.IM_HEIGHT, self.IM_WIDTH),
                                                                     batch_size=self.batch_size)

        LOGGER.info('Saving train generator class indices...')
        with open(os.path.join(self.training_essentials_folder, 'class_indices.json'), 'w') as fp:
            json.dump(self.train_generator.class_indices, fp)


    def __transfer_learn_on_model(self):
        learning_rate = self.tl_lr

        LOGGER.info('Compiling Base Model for transfer learning.')
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate),
                           metrics=['acc'])

        LOGGER.info('Saving Model Architecture as JSON.')
        model_as_json = self.model.to_json()
        with open(os.path.join(self.training_essentials_folder, 'transfer_learning_model_architecture.json'), 'w') as f:
            f.write(model_as_json)

        history = self.model.fit_generator(self.train_generator,
                                          steps_per_epoch=self.steps_per_epoch,
                                          epochs=self.tl_epochs,
                                          validation_data=self.validation_generator,
                                          validation_steps=self.validation_steps,
                                          callbacks=[self.tensorboard, self.custom_logger,
                                                     CustomModelCheckpoint(self.model, os.path.join(self.model_directory,
                                                                                                    self.tl_model_name))],
                                          workers=self.cpu_count)

        LOGGER.info('Model fitting completed for transfer learning.')
        LOGGER.info('Saving model weights.')
        self.model.save_weights(
            os.path.join(self.model_directory, 'transfer_learning_model_weights.h5'))
        LOGGER.info('Saving model.')
        self.model.save(os.path.join(self.model_directory, 'transfer_learning_model.h5'))


    def __fine_tune_on_model(self):
        learning_rate = self.ft_lr

        LOGGER.info('Compiling Base Model for fine tuning.')
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate),
                           metrics=['acc'])

        LOGGER.info('Saving Model Architecture as JSON.')
        model_as_json = self.model.to_json()
        with open(os.path.join(self.training_essentials_folder, 'fine_tuning_model_architecture.json'), 'w') as f:
            f.write(model_as_json)

        history = self.model.fit_generator(self.train_generator,
                                          steps_per_epoch=self.steps_per_epoch,
                                          epochs=self.ft_epochs,
                                          validation_data=self.validation_generator,
                                          validation_steps=self.validation_steps,
                                          callbacks=[self.tensorboard, self.custom_logger,
                                                     CustomModelCheckpoint(self.model, os.path.join(self.model_directory,
                                                                                                    self.ft_model_name))],
                                          workers=self.cpu_count)

        LOGGER.info('Model fitting completed for fine tuning.')
        LOGGER.info('Saving model weights.')
        self.model.save_weights(
            os.path.join(self.model_directory, 'fine_tuning_model_weights.h5'))
        LOGGER.info('Saving model.')
        self.model.save(os.path.join(self.model_directory, 'fine_tuning_model.h5'))



    def __transfer_learn_on_parallel_model(self, model):
        learning_rate = self.tl_lr

        LOGGER.info('Compiling Base Model for transfer learning.')
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate),
                           metrics=['acc'])

        LOGGER.info('Compiling Parallel Model for transfer learning.')
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['acc'])

        LOGGER.info('Saving Model Architecture as JSON.')
        model_as_json = self.model.to_json()
        with open(os.path.join(self.training_essentials_folder, 'transfer_learning_model_architecture.json'), 'w') as f:
            f.write(model_as_json)

        history = model.fit_generator(self.train_generator,
                                      steps_per_epoch=self.steps_per_epoch,
                                      epochs=self.tl_epochs,
                                      validation_data=self.validation_generator,
                                      validation_steps=self.validation_steps,
                                      callbacks=[self.tensorboard, self.custom_logger,
                                                 CustomModelCheckpoint(self.model, os.path.join(self.model_directory,
                                                                                                self.tl_model_name))],
                                      workers=self.cpu_count)
        LOGGER.info('Model fitting completed for transfer learning.')
        LOGGER.info('Saving model weights.')
        self.model.save_weights(
            os.path.join(self.model_directory, 'transfer_learning_model_weights.h5'))
        LOGGER.info('Saving model.')
        self.model.save(os.path.join(self.model_directory, 'transfer_learning_model.h5'))



    def __fine_tune_on_parallel_model(self, model):
        learning_rate = self.ft_lr

        LOGGER.info('Compiling Base Model for fine tuning.')
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate),
                           metrics=['acc'])

        LOGGER.info('Compiling Parallel Model for fine tuning.')
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['acc'])

        LOGGER.info('Saving Model Architecture as JSON.')
        model_as_json = self.model.to_json()
        with open(os.path.join(self.training_essentials_folder, 'fine_tuning_model_architecture.json'), 'w') as f:
            f.write(model_as_json)

        history = model.fit_generator(self.train_generator,
                                      steps_per_epoch=self.steps_per_epoch,
                                      epochs=self.ft_epochs,
                                      validation_data=self.validation_generator,
                                      validation_steps=self.validation_steps,
                                      callbacks=[self.tensorboard, self.custom_logger,
                                                 CustomModelCheckpoint(self.model, os.path.join(self.model_directory,
                                                                                                self.ft_model_name))],
                                      workers=self.cpu_count)

        LOGGER.info('Model fitting completed for fine tuning.')
        LOGGER.info('Saving model weights.')
        self.model.save_weights(
            os.path.join(self.model_directory, 'fine_tuning_model_weights.h5'))
        LOGGER.info('Saving model.')
        self.model.save(os.path.join(self.model_directory, 'fine_tuning_model.h5'))


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

        epochs = range(1, len(acc) + 1)

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
        if transfer_learn:
            LOGGER.info('Freezing Conv Base for Transfer Learning.')
            model = self.__freeze_conv_base(model)
        return model


    def begin_training(self):
        # calculate training essentials
        LOGGER.info('Calulating num_classes, samples, creating model directory, etc...')
        self.__training_essentials()
        self.__create_generators()

        train_earlier_model = config['MODELLING']['load_weights_of_previous_model']
        if train_earlier_model.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
            train_earlier_model = True
        else:
            train_earlier_model = False


        check_tl = config['MODELLING']['train_from_scratch']
        if check_tl.lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
            check_tl = False
        else:
            check_tl = True

        only_fine_tune = False

        if self.num_gpus <= 1:
            if train_earlier_model:
                LOGGER.info('Loading earlier model.')
                self.model = load_model(config['MODELLING']['previous_model_path'])
                stop_position = config['MODELLING']['previous_model_stop_position']
                LOGGER.info('Previous model was stopped at {}'.format(stop_position))
                if stop_position == 'transfer_learning':
                    # TODO:-
                    # continue transfer learning,
                    # then unfreeze layers
                    # then fine tune
                    check_tl = True
                elif stop_position == 'fine_tuning':
                    # TODO:-
                    # do not transfer learn,
                    # unfreeze layers with respect to the config
                    # start fine tuning
                    only_fine_tune = True
                    check_tl = True
                else:
                    # TODO:-
                    # continue transfer learning,
                    # do not fine tune
                    check_tl = False
            else:
                self.model = self.__build_model(transfer_learn=check_tl)
            # transfer learn with frozen conv base
            if not only_fine_tune:
                self.__transfer_learn_on_model()
            if check_tl:
                # unfreeze conv base layers
                self.__unfreeze_layers_in_model()
                # fine tune with unfrozen layers
                self.__fine_tune_on_model()
        else:
            with tf.device("/cpu:0"):
                # self.model, self.conv_base = self.__build_model(transfer_learn=check_tl)
                if train_earlier_model:
                    LOGGER.info('Loading earlier model.')
                    self.model = load_model(config['MODELLING']['previous_model_path'])
                    stop_position = config['MODELLING']['previous_model_stop_position']
                    LOGGER.info('Previous model was stopped at {}'.format(stop_position))
                    if stop_position == 'transfer_learning':
                        # TODO:-
                        # continue transfer learning,
                        # then unfreeze layers
                        # then fine tune
                        check_tl = True
                    elif stop_position == 'fine_tuning':
                        # TODO:-
                        # do not transfer learn,
                        # unfreeze layers with respect to the config
                        # start fine tuning
                        only_fine_tune = True
                        check_tl = True
                    else:
                        # continue transfer learning,
                        # do not fine tune
                        check_tl = False
                else:
                    self.model = self.__build_model(transfer_learn=check_tl)
            parallel_model = multi_gpu_model(self.model, gpus=self.num_gpus)
            # transfer learn with frozen conv base
            if not only_fine_tune:
                self.__transfer_learn_on_parallel_model(parallel_model)
            if check_tl:
                # unfreeze conv base layers
                self.__unfreeze_layers_in_model()
                # # fine tune with unfrozen layers
                self.__fine_tune_on_parallel_model(parallel_model)



if __name__ == '__main__':
    try:
        tl_instance = TransferLearning(train_dir=config['FILEPATHS']['train_dir'],
                                       val_dir=config['FILEPATHS']['val_dir'])
        LOGGER.info('Beginning Training Process')
        tl_instance.begin_training()
    except Exception as e:
        LOGGER.error(e, exc_info=True)