import collections
import os
import argparse, sys
import numpy as np
import pandas as pd
import scipy as sp
import pickle

import keras
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.layers import merge, Embedding, Bidirectional
from keras.layers.core import *
from keras.models import *
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import *
from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import scipy as sp

#load data
with open('data/esp_seq_data_array.pkl', 'rb') as handle:
    esp_data = pickle.load(handle)
with open('data/hf_seq_data_array.pkl', 'rb') as handle:
    hf_data = pickle.load(handle)
    

def load_data(X,X_biofeat,y, test_size = 0.15,random_state=40):
    X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=test_size, random_state=random_state)

    X_train_biofeat, X_test_biofeat, y_train, y_test = train_test_split(
       X_biofeat, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, X_train_biofeat, X_test_biofeat, y_train, y_test

def get_metrics(model,model_type='esp'):
    if model_type == 'esp':
        X,X_biofeat,y = esp_data
    elif model_type == 'hf':
        X,X_biofeat,y = hf_data
    X_train, X_test, X_train_biofeat, X_test_biofeat, y_train, y_test = load_data(X, X_biofeat, y,random_state=40) 
    y_train_pred = model.predict([X_train,X_train_biofeat])
    y_test_pred = model.predict([X_test,X_test_biofeat])
    mse = mean_squared_error( y_test, y_test_pred)
    spearmanr = sp.stats.spearmanr(y_test, y_test_pred)[0]    
    return 'MES:' + str(mse),'Spearman:' + str(spearmanr)

def lstm_model(model_type='esp', batch_size=90, epochs=50, initializer='0',em_dim=44,em_drop=0.2,
                rnn_units=60, rnn_drop=0.6, rnn_rec_drop=0.1, fc_num_hidden_layers=3,
                fc_num_units=320, fc_drop=0.4,fc_activation='elu',optimizer=Adam,learning_rate=0.001,
                validation_split=0.1,shuffle=False):
    if model_type == 'esp':
        X,X_biofeat,y = esp_data
    elif model_type == 'hf':
        X,X_biofeat,y = hf_data
    X_train, X_test, X_train_biofeat, X_test_biofeat, y_train, y_test = load_data(X, X_biofeat, y,random_state=33) 
    
    fc_activation = fc_activation_dict[str(fc_activation)]
    initializer = initializer_dict[str(initializer)]
    optimizer = optimizer_dict[str(optimizer)]
    sequence_input = Input(name = 'seq_input', shape = (22,))

    embedding_layer = Embedding(7,em_dim,input_length=22)
    embedded = embedding_layer(sequence_input)
    embedded = SpatialDropout1D(em_drop)(embedded)
    x = embedded

    #RNN
    lstm = LSTM(rnn_units, dropout=rnn_drop, 
                kernel_regularizer='l2',recurrent_regularizer='l2',
                recurrent_dropout=rnn_rec_drop, return_sequences=True)
    x = Bidirectional(lstm)(x)
    x = Flatten()(x)

    #biological featues
    biological_input = Input(name = 'bio_input', shape = (X_train_biofeat.shape[1],))
    x = keras.layers.concatenate([x, biological_input])


    for l in range(fc_num_hidden_layers):
        x = Dense(fc_num_units, activation=fc_activation)(x)
        x = Dropout(fc_drop)(x)
    #finish model
    mix_output = Dense(1, activation='linear',name='mix_output')(x)

    model = Model(inputs=[sequence_input, biological_input], outputs=[mix_output])
    #model = Model(inputs=[sequence_input], outputs=[mix_output])
    model.compile(loss='mse', optimizer=optimizer(lr=0.001))
    
    np.random.seed(1337)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    get_best_model = GetBest('models/' + model_type + '_rnn.hd5',monitor='val_loss', verbose=1, mode='min')
    model.fit([X_train,X_train_biofeat], 
    #model.fit([X_train], 
                 y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2,
                 validation_split=0.1,
                 shuffle=False,
                 callbacks=[get_best_model, early_stopping])    
    return model

fc_activation_dict = {'1':'relu','2':'tanh', '3':'sigmoid', '4':'hard_sigmoid', '0':'elu'}
initializer_dict = {'1':'lecun_uniform','2':'normal', '3':'he_normal', '0':'he_uniform'}
optimizer_dict = {'1':SGD,'2':RMSprop, '3':Adagrad, '4':Adadelta,'5':Adam,'6':Adamax,'0':Nadam}

import numpy as np
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
            
class GetBest(Callback):
    def __init__(self,filepath=None, monitor='val_loss', save_best=False,verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.save_best = save_best
        self.filepath = filepath
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
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
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                    #self.model.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve.' %
                              (epoch + 1, self.monitor)) 
                    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f.' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)
        #self.model.save(self.filepath, overwrite=True)