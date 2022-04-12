# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:27:03 2021

@author: bjpsa
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json
from Vocabulary2 import Vocabulary
from predictor import Predictor
from WGAN_4 import WGANGP
from Autoencoder2_emb import Autoencoder as AE
from utils import *
import time
from csv import reader
TF = False

#criar folders
path = os.getcwd()
n_run = 'run_feedback6' #update 20% and new dataset
RUN_FOLDER = path+'/GAN_model/{}/'.format(n_run)
#RUN_FOLDER = path+'\\GAN_model\\{}\\'.format(n_run)
if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'generated_data'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
    os.mkdir(os.path.join(RUN_FOLDER, 'feedbackGAN'))
    #os.mkdir(os.path.join(RUN_FOLDER, 'TF'))

#load data
path = 'data/'
filename = 'data_clean_kop.csv'
file_path = path + filename

config_file = 'configReinforce.json' # Configuration file
property_identifier = 'kor' # It can be 'a2d', kor', 'qed', 'sas', 'logP', or 'jak2'
configReinforce=load_config(config_file, property_identifier)
smiles_raw, labels_raw= reading_csv(
        configReinforce, property_identifier)

#Configuration for predictor
config_file_predictor = 'configPredictor.json'
configReinforce=load_config(config_file_predictor, property_identifier)

# Create Vocabulary object
vocab = Vocabulary('Vocab_complete.txt',max_len = 70) #longest smile has 65

#Preprocess data
tok, _ = vocab.tokenize(smiles_raw)

#model2
X_train = vocab.encode(tok)


n=len(X_train)

X_train = np.reshape(X_train, (n, vocab.max_len,1))#(1000, 100, 1)

X2_train = vocab.one_hot_encoder(tok)

decoder_input_shape = X2_train.shape[1:] #(max_len, vocab.size)
output_dim = vocab.vocab_size # = vocab.siz~e
print(decoder_input_shape)

#Load trained autoencoder
path_model = 'AE/Exp9model2256_500000_biLSTM2_units512_dec_layers2-128-0.9-adam-0.1-256/'
#decoder_input_shape = 
latent_dim = 256
lstm_units = 512
batch_norm = True
batch_norm_momentum = 0.9
noise_std = 0.1
numb_dec_layer = 2 # = n_bLSTM
emb_dim = 256
decoder_input_shape = (vocab.max_len, vocab.vocab_size)
output_dim = vocab.vocab_size
autoencoder = AE(path_model, decoder_input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, emb_dim, vocab.vocab_size, vocab.max_len)

autoencoder.load_autoencoder_model(path_model+'model--86--0.0013.hdf5')

#pass the data through the autoencoder
x_latent = autoencoder.smiles_to_latent_model.predict(X_train) #---> this will be the real data to train the GAN

# Load Predictor Object
predictor = Predictor(configReinforce, vocab, 'dnn', 'SMILES', property_identifier, False)

batch_size = 64
data = x_latent

#create GAN
input_dim = latent_dim
critic_layers_units = [256,256,256]
critic_lr = 0.0001#0.0002
gp_weight = 10
z_dim  = 64        #### try dif values
generator_layers_units = [128,256,256,256,256]
generator_batch_norm_momentum = 0.9
generator_lr = 0.0001
batch_size = 64
critic_optimizer = 'adam'
generator_optimizer = 'adam'
critic_dropout = 0.2
generator_dropout = 0.2

gan = WGANGP(input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout,batch_size, critic_optimizer, generator_optimizer)

#Load Trained GAN
weights_critic_filepath = 'GAN_model/GAN_weights/critic_weights-10000.h5'
weights_generator_filepath = 'GAN_model/GAN_weights/generator_weights-10000.h5'
gan.load_weights(weights_critic_filepath,weights_generator_filepath)


#if TF == True:
#    # Transfer Learning
#    train_start = process_time()
#    gan.train(data, batch_size, epochs, RUN_FOLDER+'TF/', autoencoder, vocab, print_every_n_epochs = 250, critic_loops = 5)
#    train_time = process_time()-train_start

run_folder = 'GAN_model/'+n_run+'/'

#Evaluate GAN before applying feedback loop
gan.epoch = 0
gan.autoencoder = autoencoder
gan.vocab = vocab

valid_smiles_before, perc_valid_before, _ = gan.sample_valid_data(10, run_folder, True)
predictions_before, _ = evaluate_property(predictor, valid_smiles_before, property_identifier)

list_predictions_before=predictions_before.tolist()
with open(os.path.join(run_folder, "prediction_before.csv"), 'w') as f:
    writer = csv.writer(f)
    for i in range(len(list_predictions_before)):
        writer.writerow([list_predictions_before[i]])

print('start feedback gan..........................')

n_to_generate = 20
threshold = 5
info = 'max'
epochs =1

gan.train_feedbackGAN(data, smiles_raw, batch_size, epochs, run_folder, autoencoder, vocab, predictor, n_to_generate, threshold, info, property_identifier, print_every_n_epochs = 50, critic_loops = 5)

valid_smiles_after, perc_valid_after, _ = gan.sample_valid_data(10, run_folder, True)
predictions_after, _ = evaluate_property(predictor, valid_smiles_after, property_identifier)


print('Percentage of valid smiles (after feedbackGAN): ', perc_valid_after)

#Write predicitions

list_predictions_before=predictions_before.tolist()



list_predictions_after=predictions_after.tolist()
with open(os.path.join(run_folder, "prediction_after.csv"), 'w') as f:
    writer = csv.writer(f)
    for i in range(len(list_predictions_after)):
        writer.writerow([list_predictions_after[i]])


#Retrieve predictions before from csv

#predictions_before=[]
#with open("prediction_before.csv","r") as read_obj:
#
#    csv_reader=reader(read_obj)
#    for row in csv_reader:
#        predictions_before.append(row[0])

predictions_before=np.array(predictions_before).astype(np.float)
print(predictions_after)
print(type(predictions_after))
diff = plot_hist_both(predictions_before,predictions_after, property_identifier)
print('difference between the averages of the predicted properties:', diff)


with open(os.path.join(run_folder, "testing_500_epochs.csv"), 'w') as f:
    writer = csv.writer(f)
    for i in range(len(valid_smiles_after)):
        writer.writerow(valid_smiles_after[i])
