import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import models, layers, optimizers
import glob
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard
import time
import multiprocessing
import json


class TransferLearning(object):

    def __init__(self, train_dir, val_dir):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.datagen = ImageDataGenerator(rescale=1./255)
        self.batch_size = 64
        self.IM_WIDTH = 299
        self.IM_HEIGHT = 299
        self.epochs = 30
        self.model_driectory = 'models'
        self.tensorboard_logs_dir = 'tensorboard_logs'
        self.model_name = 'tl_dog_breed_inceptionv3.{epoch:02d}-{val_loss:.2f}.hdf5'
        self.tensorboard_logs_name = "tl_dog_breed_inceptionv3_{}".format(time.time())
        self.cpu_count = multiprocessing.cpu_count()
        self.fraction_to_unfreeze_while_fine_tuning = 0.4
        self.checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_driectory,
                                                           self.model_name),
                                          monitor='val_loss',
                                          save_best_only=True)

        self.tensorboard = TensorBoard(
            log_dir=os.path.join(self.tensorboard_logs_dir, self.tensorboard_logs_name))


    def __unfreeze_layers_in_model(self):
        self.conv_base.trainable = True

        num_layers = len(self.conv_base.layers)
        NB_IV3_LAYERS_TO_FREEZE = num_layers - int(self.fraction_to_unfreeze_while_fine_tuning * num_layers)

        for layer in self.conv_base.layers[:NB_IV3_LAYERS_TO_FREEZE]:
            layer.trainable = False
        for layer in self.conv_base.layers[NB_IV3_LAYERS_TO_FREEZE:]:
            layer.trainable = True

        print('Number of trainable weights after unfreezing the conv base: {}'.format(len(self.conv_base.trainable_weights)))


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


    def __create_conv_base(self):
        self.conv_base = InceptionV3(weights='imagenet',
                                     include_top=False,
                                     input_shape=(self.IM_HEIGHT, self.IM_WIDTH, 3))
        # self.conv_base.summary()


    def __freeze_conv_base(self):
        print('Number of trainable weights before freezing the conv base: {}'.format(len(self.model.trainable_weights)))
        self.conv_base.trainable = False
        print('Number of trainable weights after freezing the conv base: {}'.format(len(self.model.trainable_weights)))


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

        print('Saving train generator class indices...')
        with open('train_class_indices.json', 'w') as fp:
            json.dump(self.train_generator.class_indices, fp)

        print('Saving val generator class indices...')
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

        history = self.model.fit_generator(self.train_generator,
                                           steps_per_epoch=self.steps_per_epoch,
                                           epochs=self.epochs,
                                           validation_data=self.validation_generator,
                                           validation_steps=self.validation_steps,
                                           callbacks=[self.checkpoint, self.tensorboard],
                                           workers=self.cpu_count)

        # self.plot(history)




    def __create_model(self):
        self.model = models.Sequential()
        self.model.add(self.conv_base)
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dense(1024, activation='relu'))
        self.model.add(layers.Dense(self.NUM_CLASSES, activation='softmax'))
        self.model.summary()


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


    def begin_training(self):
        # calculate training essentials
        print('Calulating num_classes, samples, creating model directory, etc...')
        self.__training_essentials()
        # create conv base
        print('Creating Inception V3 conv base')
        self.__create_conv_base()
        # create model with conv_base
        self.__create_model()
        # freeze conv base layers
        self.__freeze_conv_base()
        # transfer learn with frozen conv base
        self.__create_generators()
        self.__train_model()
        # unfreeze conv base layers
        self.__unfreeze_layers_in_model()
        # fine tune with unfrozen layers
        self.__train_model(fine_tuning=True)



if __name__ == '__main__':
    tl_instance = TransferLearning(train_dir='D:\\Python Projects\\Projects\\Dog Breed Identification\\TransferLearning\\dog_breeds_train_val_split\\train',
                                   val_dir='D:\\Python Projects\\Projects\\Dog Breed Identification\\TransferLearning\\dog_breeds_train_val_split\\val')
    tl_instance.begin_training()