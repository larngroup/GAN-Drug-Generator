# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:01:59 2021

@author: bjpsa
"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, Input, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
from tokens import tokens_table

from utils import regression_plot, denormalization_with_labels, reading_csv, transform_to_array
from utils import mse, r_square,rmse,ccc
import time

class Predictor():
    def __init__(self, config, vocab, model_type, descriptor_type, property_identifier, load):
        self.dropout=0.3
        self.path_predictor="predictor_weigths_"+property_identifier+"/"+property_identifier
        token_table = tokens_table()
        self.n_table=token_table.table_len
        self.n_units=128
        config.input_length=70
        self.activation_rnn="relu"
        self.activation_dense="linear"
        self.config = config
        self.vocab = vocab
        self.model_type = model_type
        self.descriptor_type = descriptor_type  #context vector or smiles
        self.property_identifier = property_identifier # for example: kor
        self.load = load  #if False then the model is built, if True then the model is loaded
        
        self.load_models()
        _,self.labels = reading_csv(config,property_identifier)
    
    def build_model(self):
        

        my_model = Sequential()
        my_model.add(Input(shape=(self.config.input_length,)))
        my_model.add(Embedding(self.n_table, self.n_units,
                                 input_length=self.config.input_length))
        my_model.add(LSTM(self.n_units, return_sequences=True, input_shape=(
            None, self.n_units, self.config.input_length), dropout=self.dropout))
        my_model.add(LSTM(self.n_units, dropout=self.dropout))

        my_model.add(Dense(self.n_units, activation=self.activation_rnn))
        my_model.add(Dense(1, activation=self.config.activation_dense))

        return my_model
        
            
    def train_model(self, data, searchParam):
        self.data = data
        self.searchParam = searchParam
        X_train = self.data[0]
        y_train = self.data[1]
        X_val = self.data[4]
        y_val = self.data[5]
        
        opt = Adam(lr=self.config.lr, beta_1 =self.config.beta_1, beta_2 = self.config.beta_2, amsgrad = False)
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15, restore_best_weights=True)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        
        #self.model.compile(loss=self.config.loss_criterium, optimizer = opt,  metrics=['accuracy','AUC',specificity,sensitivity,matthews_correlation])
        #self.model.compile(loss='r', optimizer = opt,  metrics=['accuracy',specificity,sensitivity,matthews_correlation])
        self.model.compile(loss=self.config.loss_criterium, optimizer = opt, metrics=[mse, r_square, rmse, ccc])
        
        result = self.model.fit(X_train, y_train, epochs = self.config.epochs_dense, 
        #                         batch_size = self.config.batch_size, 
        #                         validation_data=(X_val, y_val), callbacks = [es, mc])
        #result = self.model.fit(X_train, y_train, epochs = 100, 
                                  batch_size = self.config.batch_size, 
                                  validation_data=(X_val, y_val), callbacks = [es, mc])
        #-----------------------------------------------------------------------------
        # Plot learning curves including R^2 and RMSE
        #-----------------------------------------------------------------------------
        
        # # plot training curve for R^2 (beware of scale, starts very low negative)
        # plt.plot(result.history['accuracy'])
        # plt.plot(result.history['val_accuracy'])
        # plt.title('model Accuracy')
        # plt.ylabel('Acc')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
                   
        # # plot training curve for rmse
        # plt.plot(result.history['specificity'])
        # plt.plot(result.history['val_specificity'])
        # plt.title('specificity')
        # plt.ylabel('spec')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        
        # # plot training curve for rmse
        # plt.plot(result.history['sensitivity'])
        # plt.plot(result.history['val_sensitivity'])
        # plt.title('sensitivity')
        # plt.ylabel('sensi')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        
        #-----------------------------------------------------------------------------
        # Plot learning curves including R^2 and RMSE
        #-----------------------------------------------------------------------------
        
        # plot training curve for R^2 (beware of scale, starts very low negative)
        plt.plot(result.history['r_square'])
        plt.plot(result.history['val_r_square'])
        plt.title('model R^2')
        plt.ylabel('R^2')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
                   
        # plot training curve for rmse
        plt.plot(result.history['rmse'])
        plt.plot(result.history['val_rmse'])
        plt.title('rmse')
        plt.ylabel('rmse')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        
        
        if self.searchParam:
            print('\nEvaluating on the test set')
            metrics = self.model.evaluate(x = self.data[2], y = self.data[3])        #test 
            print("\n\nMean_squared_error: ",metrics[1],"\nR_square: ", metrics[2], "\nRoot mean square: ",metrics[3], "\nCCC: ",metrics[4])
            print(metrics)
            
            if self.descriptor_type == 'SMILES':
                values= [self.config.dropout,self.config.batch_size,self.config.lr,self.config.n_units,
                 self.config.rnn,self.config.activation_rnn,self.config.epochs,metrics[0],metrics[1],metrics[2],metrics[3]]
                file=[i.rstrip().split(',') for i in open('grid_results_'+self.descriptor_type+'.csv').readlines()]
                file.append(values)
                file=pd.DataFrame(file)
                file.to_csv('grid_results_'+self.descriptor_type+'.csv',header=None,index=None)
            elif self.descriptor_type == 'context_vector':
                values= [self.config.dropout_dense,self.config.batch_size,self.config.lr,self.config.n_units,
                 self.config.activation_dense,self.config.epochs,metrics[0],metrics[1],metrics[2],metrics[3], metrics[4]] 
                with open('grid_results_'+self.descriptor_type+'3.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(values)

        else:
            filepath=""+self.config.checkpoint_dir + ""+ self.config.model_name
            #serialize model to JSON
            model_json = self.model.to_json()
            with open(str(filepath+self.descriptor_type + ".json"), "w") as json_file:
                json_file.write(model_json)
                # serialize weights to HDF5
            self.model.save_weights(str(filepath + ".h5"))
            print("Saved model to disk")
    
    
    def load_models(self):
        
        loaded_models = []

        for i in range(5):
            my_model=self.build_model()
            my_model.load_weights(self.path_predictor+"model"+str(i)+".hdf5")


            print("Model " + str(i) + " loaded from disk!")
            loaded_models.append(my_model)

        self.loaded_models = loaded_models



     
    def evaluator(self,data):
       """
       This function evaluates the QSAR models previously trained
       ----------
       data: List with testing SMILES and testing labels
       Returns
       -------
       This function evaluates the model with the training data
       """
       
       print("\n------- Evaluation with test set -------")
       smiles = tf.convert_to_tensor(data[2], np.float32)
       label = tf.convert_to_tensor(data[3], np.float32)
       metrics = []
       prediction = []
       opt = Adam(lr=self.config.lr, beta_1=self.config.beta_1, beta_2=self.config.beta_2, amsgrad=False)
       for m in range(len(self.loaded_models)):
           #self.loaded_models[m].compile(loss=self.config.loss_criterium, optimizer = opt,  metrics=['accuracy','AUC',specificity,sensitivity,matthews_correlation])
           self.loaded_models[m].compile(loss=self.config.loss_criterium,optimizer = opt,  metrics=[mse,r_square,rmse,ccc])
           
           metrics.append(self.loaded_models[m].evaluate(x = smiles, y = label))
           print('\n\nmetrics: ', metrics)
           
           prediction.append(self.loaded_models[m].predict(smiles))
       #print(prediction)
       prediction = np.array(prediction).reshape(len(self.loaded_models), -1)
       prediction = np.mean(prediction, axis = 0)
       
       regression_plot(label, prediction)
       
       if self.model_type == 'dnn':
           metrics = np.array(metrics).reshape(len(self.loaded_models), -1)
           metrics = metrics[:,1:]
           
       metrics = np.mean(metrics, axis = 0)
     
       return metrics, label, prediction
   
    def predict(self, smiles):
        #loads the models and predicts for new smiles
        #smiles must already be encoded
        
        prediction = []
        print('len(smiles) to rpedict : ', len(smiles))        
        #smiles = transform_to_array(np.asarray(self.vocab.encode(self.vocab.tokenize(smiles))))
        smiles_tok, og_idx = self.vocab.tokenize(smiles)
        #smiles = np.asarray(self.vocab.encode(self.vocab.tokenize(smiles)))
        smiles = np.asarray(self.vocab.encode(smiles_tok))
        print('array smiles: ',smiles.shape)
        #print(smiles)
        for m in range(len(self.loaded_models)):
            prediction.append(self.loaded_models[m].predict(smiles))
                
        prediction = np.array(prediction).reshape(len(self.loaded_models), -1)
        #print(self.labels)
        #prediction = denormalization(prediction,data)
        prediction = denormalization_with_labels(prediction,self.labels)
        
        prediction = np.mean(prediction, axis = 0)
       
        return prediction, og_idx
