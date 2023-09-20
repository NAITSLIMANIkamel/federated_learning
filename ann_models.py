# Copyright (c) 2023 @ FBK - Fondazione Bruno Kessler
# Author: Roberto Doriguzzi-Corin
# Project: FLAD, Adaptive Federated Learning for DDoS Attack Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set tensorflow log level
from util_functions import *
import tensorflow as tf
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input, Dense, Activation,  Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Model, Sequential, save_model, load_model, clone_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D
from keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow._api.v2.math import reduce_sum, square

K.set_image_data_format('channels_last')

# disable GPUs for test reproducibility
tf.config.set_visible_devices([], 'GPU')

KERNELS = 256
MLP_UNITS = 32

MIN_EPOCHS = 1
MAX_EPOCHS = 5
MIN_STEPS = 10
MAX_STEPS = 1000

def compileModel(model, optimizer_type="SGD",loss='binary_crossentropy'):
    if optimizer_type == "Adam":
        optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    else:
        optimizer = SGD(learning_rate=0.1, momentum=0.0, nesterov=False)

    model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])

# Convolutional NN
def RNNModel(input_shape=122):
    K.clear_session()

        #Bidirectional RNN
    model_rnn = Sequential()
    model_rnn.add(Convolution1D(64, kernel_size=50,activation="relu",input_shape=(input_shape, 1)))
    model_rnn.add(MaxPooling1D(pool_size=(1)))
    model_rnn.add(BatchNormalization())
    model_rnn.add(Bidirectional(LSTM(64, return_sequences=False))) 
    model_rnn.add(Reshape((128, 1), input_shape = (128, )))
    
    model_rnn.add(MaxPooling1D(pool_size=(1)))
    model_rnn.add(BatchNormalization())
    model_rnn.add(Bidirectional(LSTM(128, return_sequences=False))) 
    
    model_rnn.add(Dropout(0.5))
    model_rnn.add(Dense(5))
    model_rnn.add(Activation('softmax'))
    model_rnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model_rnn

# MPL model
def MLPModel(hidden_layer=64):
    K.clear_session()
    model_mlp= MLPClassifier(hidden_layer)
    
    return model_mlp

# KNN model 
def KNNModel(neighbors=5):
    model_knn =  KNeighborsClassifier(n_neighbors=neighbors)

    return model_knn

def SVMModel(gamma="auto"):
    model_svm =  SVC(gamma=gamma)

    return model_svm

def init_server(model_type, dataset_name, input_shape, max_flow_len):
    server = {}
    server['name'] = "Server"
    features = input_shape[1]

    if model_type == 'rnn':
        server['model'] = RNNModel('rnn', input_shape, kernels=KERNELS, kernel_rows=min(3,max_flow_len), kernel_col=features)
    elif model_type == 'mlp':
        server['model'] = MLPModel('mlp', input_shape, units=MLP_UNITS)
    elif model_type == "knn" :
        server['model'] = MLPModel('knn', input_shape, units=MLP_UNITS)
    elif  model_type == "svm" :
        server['model'] = MLPModel('svm', input_shape, units=MLP_UNITS) 

    elif model_type is not None:
        try:
            print ("Loading model from file: ", model_type)
            server['model'] = load_model(model_type,compile=False)
        except:
            print("Error: Invalid model file!")
            print("Initialising an MLP as the primary global model...")
            server['model'] = FCModel('mlp', input_shape, units=MLP_UNITS)
    else:
        print("Error: Please use option \"model\" to indicate a model type (mlp or cnn), or to provide a pretrained model in h5 format")
        return None

    return server
    

def init_client(subfolder, X_train, Y_train, X_val, Y_val, dataset_name, time_window, max_flow_len):
    client = {}
    client['name'] = subfolder.strip('/').split('/')[-1] #name of the client based on the folder name
    client['folder'] = subfolder
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    client['training'] = (X_train_tensor,Y_train)
    X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
    client['validation'] = (X_val_tensor,Y_val)
    client['samples'] = client['training'][1].shape[0]
    client['dataset_name'] = dataset_name
    client['input_shape'] = client['training'][0].shape[1:4]
    client['features'] = client['training'][0].shape[2]
    client['classes'] =  np.unique(Y_train)
    client['time_window'] = time_window
    client['max_flow_len'] = max_flow_len
    client['flddos_lambda'] = 0.9 if "WebDDoS" in client['name'] or "Syn" in client['name'] else 1
    reset_client(client)
    return client

def reset_client(client):
    client['local_model'] = None # local model trained only with local data (FLDDoS comparison)
    client['f1_val'] = 0         # F1 Score of the current global model on the validation set
    client['f1_val_best'] = 0    # F1 Score of the best model on the validation set
    client['loss_train'] = float('inf')
    client['loss_val'] = float('inf')
    client['epochs'] = MIN_EPOCHS
    client['steps_per_epoch'] = MIN_STEPS
    client['rounds'] = 0
    client['round_time'] = 0
    client['update'] = True

def check_clients(clients):
    input_shape = clients[0]['input_shape']
    features = clients[0]['features']
    classes = clients[0]['classes']
    time_window = clients[0]['time_window']
    max_flow_len = clients[0]['max_flow_len']
    for client in clients:
        if input_shape != client['input_shape'] or \
            features != client['features'] or \
            classes.all() != client['classes'].all() or \
            time_window != client['time_window'] or \
            max_flow_len != client['max_flow_len']:
                print("Inconsistent clients properties!")
                return False
    return True