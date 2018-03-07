import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import models, layers, optimizers
import glob
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard
import time


class FeatureExtractor(object):

    def __init__(self, train_dir, val_dir):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.datagen = ImageDataGenerator(rescale=1./255)
        self.batch_size = 32
        self.IM_WIDTH = 299
        self.IM_HEIGHT = 299
        self.epochs=50
        self.model_driectory = 'models'


    def count_number_of_files(self, root_directory):
        count = 0
        for files in glob.glob(os.path.join(root_directory, '*')):
            count += len(os.listdir(files))
        return count


    def training_essentials(self):
        self.NUM_CLASSES = len(os.listdir(self.train_dir))
        self.NUM_TRAINING_SAMPLES = self.count_number_of_files(self.train_dir)
        self.NUM_VALIDATION_SAMPLES = self.count_number_of_files(self.val_dir)
        if not os.path.isdir(self.model_driectory):
            os.makedirs(self.model_driectory)


    def create_conv_base(self):
        self.conv_base = InceptionV3(weights='imagenet',
                                     include_top=False,
                                     input_shape=(self.IM_HEIGHT, self.IM_WIDTH, 3))
        self.conv_base.summary()


    def extract_features(self, directory, sample_count):
        features = np.zeros(shape=(sample_count, 8, 8, 2048))
        labels = np.zeros(shape=(sample_count, self.NUM_CLASSES))
        print('Shape of features -> {}. shape of labels -> {}'.format(features.shape, labels.shape))

        generator = self.datagen.flow_from_directory(directory=directory,
                                                     target_size=(self.IM_WIDTH, self.IM_HEIGHT),
                                                     batch_size=self.batch_size)
        i=0
        for inputs_batch, labels_batch in generator:
            features_batch = self.conv_base.predict(inputs_batch)
            features[i * self.batch_size: (i+1) * self.batch_size] = features_batch
            labels[i * self.batch_size: (i+1) * self.batch_size] = labels_batch
            i += 1
            if i * self.batch_size >= sample_count:
                break

        return features, labels


    def find_train_and_val_features(self):
        self.train_features, self.train_labels = self.extract_features(self.train_dir, self.NUM_TRAINING_SAMPLES)
        self.val_features, self.val_labels = self.extract_features(self.val_dir, self.NUM_VALIDATION_SAMPLES)

        self.train_features = np.reshape(self.train_features, (self.NUM_TRAINING_SAMPLES, 8*8*2048))
        self.val_features = np.reshape(self.val_features, (self.NUM_VALIDATION_SAMPLES, 8*8*2048))



    def create_model(self):
        model = models.Sequential()
        model.add(layers.Dense(256, activation='relu', input_dim=8*8*2048))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.NUM_CLASSES, activation='softmax'))

        model.summary()

        return model

    def plot(self, history):
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


    def train(self):
        # calculate training essentials
        print('Calulating num_classes, samples, creating model directory, etc...')
        self.training_essentials()
        # create conv base
        print('Creating Inception V3 conv base')
        self.create_conv_base()
        # extract features
        print('Extracting features...')
        self.find_train_and_val_features()
        # create model
        model = self.create_model()
        # train
        model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        print('Beginning Training...')

        checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_driectory,
                                                           'fe_dog_breed_inceptionv3_basic_dense.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                     monitor='val_loss',
                                     save_best_only=True)

        tensorboard = TensorBoard(log_dir="tensorboard_logs/fe_dog_breed_inceptionv3_basic_dense-{}".format(time.time()))


        history = model.fit(self.train_features, self.train_labels,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(self.val_features, self.val_labels),
                            callbacks=[checkpoint, tensorboard])
        # plot
        self.plot(history)



if __name__ == '__main__':
    fe_instance = FeatureExtractor(train_dir='D:\\Python Projects\\Projects\\Dog Breed Identification\\TransferLearning\\dog_breeds_train_val_split\\train',
                                   val_dir='D:\\Python Projects\\Projects\\Dog Breed Identification\\TransferLearning\\dog_breeds_train_val_split\\val')
    fe_instance.train()