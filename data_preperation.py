import os
import glob
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm


class DataPreperation(object):

    def __init__(self):
        self.X = []
        self.y = []
        self.train_dir = None
        self.val_dir = None


    def create_directories_if_do_not_exist(self, list_of_directory_paths):
        for directory in list_of_directory_paths:
            if not os.path.isdir(directory):
                os.makedirs(directory)


    def copy_files_wrt_x_and_y(self, X, y, directory):
        for idx, value in enumerate(tqdm(X)):
            y_class = y[idx]
            if not os.path.isdir(os.path.join(directory, y_class)):
                os.makedirs(os.path.join(directory, y_class))
            if not os.path.isfile(os.path.join(directory, y_class, os.path.basename(value))):
                shutil.copy(value, os.path.join(directory, y_class, os.path.basename(value)))


    def prepare_train_and_validation_split(self, input_root_path, output_root_path, ext='*.jpg', test_size=0.2):
        """
        reads data from the input root path and creates a train validation split in the output root path.
        :param input_root_path: input root folder which expects classes to be split in sub directories
        :param output_root_path: output path for creation of train, val split
        :return:
        """
        print('Finding files...')
        for class_directories in glob.glob(os.path.join(input_root_path, '*')):
            for images in glob.glob(os.path.join(class_directories, ext)):
                self.X.append(images)
                self.y.append(os.path.basename(class_directories))

        print('\nCreating train test split...')
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y)

        self.train_dir = os.path.join(output_root_path, 'train')
        self.val_dir = os.path.join(output_root_path, 'val')

        self.create_directories_if_do_not_exist([output_root_path, self.train_dir, self.val_dir])

        print('\nCopying Training files...')
        self.copy_files_wrt_x_and_y(X_train, y_train, directory=self.train_dir)

        print('\nCopying Validation files...')
        self.copy_files_wrt_x_and_y(X_test, y_test, directory=self.val_dir)

        print('Done!!')



if __name__ == '__main__':
    DataPreperation().prepare_train_and_validation_split(input_root_path='dog_breeds',
                                                         output_root_path='dog_breeds_train_val_split')