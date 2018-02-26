import configparser
import socket
import uuid

config = configparser.ConfigParser()

config['LOGGING'] = {}
config['LOGGING']['log_to_file'] = 'False'
config['LOGGING']['log_file_name'] = 'Transfer_Learning_' + socket.gethostname() + '_' + str(uuid.uuid4()) + '.log'

config['MODELLING'] = {}

config['MODELLING']['batch_size'] = "64"
config['MODELLING']['im_width'] = "299"
config['MODELLING']['im_height'] = "299"
config['MODELLING']['epochs'] = "50"
config['MODELLING']['model_directory'] = 'models'
config['MODELLING']['tensorboard_logs_dir'] = 'tensorboard_logs'
config['MODELLING']['cpu_count'] = "8"
config['MODELLING']['last_layer_fc_size'] = "1024"
config['MODELLING']['num_layers_to_freeze_while_fine_tuning'] = "172"
config['MODELLING']['num_gpus'] = "1"

config['MODEL_CHECKPOINT'] = {}

config['MODEL_CHECKPOINT']['monitor'] = "val_loss"

config['FILEPATHS'] = {}

config['FILEPATHS']['train_dir'] = 'D:\\Python Projects\\Projects\\Dog Breed Identification\\TransferLearning\\dog_breeds_train_val_split\\train'
config['FILEPATHS']['val_dir'] = 'D:\\Python Projects\\Projects\\Dog Breed Identification\\TransferLearning\\dog_breeds_train_val_split\\val'

with open('configurations.ini', 'w') as configfile:
    config.write(configfile)