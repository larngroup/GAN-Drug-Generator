# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:23:52 2020

@author: bjpsa
"""
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Concatenate,LSTM, Bidirectional, Dense, Input, GaussianNoise, BatchNormalization, RepeatVector, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, History, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
import numpy as np
import tensorflow as tf

from Vocabulary2 import Vocabulary
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

class Autoencoder:
    def __init__(self, model_path, input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer, emb_dim, vocab_size, max_len):
        
        #folder 
        self.path = model_path 
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.output_dim = output_dim
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.noise_std = noise_std

        self.numb_dec_layer = numb_dec_layer        #numero de layers no decoder

        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.max_len = max_len


        self.build_smiles_to_latent_model()
        self.build_latent_to_states_model()
        self.build_states_to_smiles_model()

        #self.build_aux_model()

        #Building the full model
        self.build_model()
        
        print(self.model.summary())
    

    def build_smiles_to_latent_model(self):

        ## 1st model

        #this model transforms smiles molecules to their latent representation

        #encoder_inputs = Input(shape = self.input_shape, name = 'encoder_inputs')
        #x = encoder_inputs
        
        #encoder_inputs = Input(shape = (self.max_len, ))
        encoder_inputs = Input(shape = (None,), name = 'encoder_inputs')
        x= Embedding(self.vocab_size, self.lstm_units//2)(encoder_inputs)
        #x = encoder_inputs
        #encoder_inputs = Embedding(self.vocab_size, self.emb_dim, input_length = self.max_len)    

        
        states_list = [] 
        states_reversed_list = []
        for i in range(self.numb_dec_layer):
            if self.numb_dec_layer == 1:
                #criar o bidirectional lstm layer
                encoder = Bidirectional(LSTM(self.lstm_units // 2, return_state = True, name = 'encoder'+str(i)+'_LSTM'))
        
                x, state_h, state_c, state_h_reverse, state_c_reverse = encoder(x)
                
                states_list.append(state_h)
                states_list.append(state_c)
                states_reversed_list.append(state_h_reverse)
                states_reversed_list.append(state_c_reverse)
                
            elif i != self.numb_dec_layer-1 :   #if it is not the last layer
                
                #criar o bidirectional lstm layer
                encoder = Bidirectional(LSTM(self.lstm_units // 2, return_sequences = True, return_state = True, name = 'encoder'+str(i)+'_LSTM'))
        
                x, state_h, state_c, state_h_reverse, state_c_reverse = encoder(x)
                
                states_list.append(state_h)
                states_list.append(state_c)
                states_reversed_list.append(state_h_reverse)
                states_reversed_list.append(state_c_reverse)
                
                if self.batch_norm:
                    x  = BatchNormalization(momentum = self.batch_norm_momentum, name = 'BN_'+str(i))(x)
                
            else:   #last layer
                
                encoder2 = Bidirectional(LSTM(self.lstm_units//2, return_state = True, name = 'encoder'+str(i)+'_LSTM'))
        
                _, state_h2, state_c2, state_h2_reverse, state_c2_reverse = encoder2(x)
                
                states_list.append(state_h2)
                states_list.append(state_c2)
                states_reversed_list.append(state_h2_reverse)
                states_reversed_list.append(state_c2_reverse)
        
        complete_states_list = states_list + states_reversed_list  #states c and h and then the reversed states
        #concatenate all states
        # states = Concatenate(axis = -1, name = 'concatenate')([state_h, state_c, state_h2, state_c2, state_h_reverse, state_c_reverse, state_h2_reverse, state_c2_reverse])
        states = Concatenate(axis = -1, name = 'concatenate')(complete_states_list)

        if self.batch_norm:
            states = BatchNormalization(momentum = self.batch_norm_momentum, name = 'BN_'+str(i+1))(states)

        #neckouptus = latent representation 
        latent_representation = Dense(self.latent_dim, activation = "relu", name = "Dense_relu_latent_rep")(states)

        if self.batch_norm:
            latent_representation = BatchNormalization(momentum = self.batch_norm_momentum, name = 'BN_latent_rep')(latent_representation)

        #Adding Gaussian Noise as a regularizing step during training
        latent_representation = GaussianNoise(self.noise_std, name = 'Gaussian_Noise')(latent_representation)


        self.smiles_to_latent_model = Model(encoder_inputs, latent_representation, name = 'smiles_to_latent_model')

        with open('smiles_to_latent.txt', 'w') as f:

            self.smiles_to_latent_model.summary(print_fn=lambda x: f.write(x + '\n'))

    def build_latent_to_states_model(self):

        ## 2nd model

        #model that, given a latent representation, constructs the initial states of the decoder


        latent_input = Input(shape =(self.latent_dim,), name = 'latent_input')

        #List that will contain the reconstructed states
        decoded_states = []

        for dec_layer in range(self.numb_dec_layer):        #2

            name = "Dense_h_" + str(dec_layer)
            h_decoder = Dense(self.lstm_units, activation = "relu", name = name)(latent_input)

            name = "Dense_c_" + str(dec_layer)
            c_decoder = Dense(self.lstm_units, activation ="relu", name = name)(latent_input)

            if self.batch_norm:
                name = "BN_h_" + str(dec_layer)
                h_decoder = BatchNormalization(momentum = self.batch_norm_momentum, name = name)(h_decoder)

                name = "BN_c_" + str(dec_layer)
                c_decoder = BatchNormalization(momentum = self.batch_norm_momentum, name = name)(c_decoder)

            decoded_states.append(h_decoder)
            decoded_states.append(c_decoder)



        self.latent_to_states_model = Model(latent_input, decoded_states, name = 'latent_to_states_model')


        with open('latent_to_states.txt', 'w') as f:

            self.latent_to_states_model.summary(print_fn=lambda x: f.write(x + '\n'))


    def build_states_to_smiles_model(self):

        ##3rd model

        #model that, given the states, outputs the probabilities for the smiles' characters    (?)

        #decoder inputs needed for teacher's forcing
        decoder_inputs = Input(shape = self.input_shape, name = "decoder_inputs")    #input_shape = decoder_input_shape

        inputs = [] #list that will have all the inputs to this model: decoder_inputs + reconstructed states (from 2nd model)

        inputs.append(decoder_inputs)

        x = decoder_inputs

        for dec_layer in range(self.numb_dec_layer): 
            name = "Decoded_state_h_" + str(dec_layer)
            state_h = Input(shape = [self.lstm_units], name = name)
            inputs.append(state_h)

            name = "Decoded_state_c_" + str(dec_layer)
            state_c = Input(shape = [self.lstm_units], name = name)
            inputs.append(state_c)

            #LSTM layer
            decoder_lstm = LSTM(self.lstm_units, return_sequences = True, name = "Decoder_LSTM_" + str(dec_layer))

            x = decoder_lstm(x, initial_state = [state_h, state_c])

            if self.batch_norm:
                x = BatchNormalization(momentum = self.batch_norm_momentum, name = "BN_decoder_"+str(dec_layer))(x)



            ################### if smt >0 Time Distributed


        #Dense layer that will return probabilities
        outputs = Dense(self.output_dim, activation = "softmax", name = "Decoder_Dense")(x)

        self.states_to_smiles_model = Model(inputs = inputs, outputs = [outputs], name = "states_to_smiles_model")
        with open('states_to_smiles.txt', 'w') as f:

            self.states_to_smiles_model.summary(print_fn=lambda x: f.write(x + '\n'))

    def build_model(self):
        encoder_inputs = Input(shape = (None,), name = "encoder_inputs") # same as the smiles_to_latent_model input
        decoder_inputs = Input(shape = self.input_shape, name = "decoder_inputs") # same as the input to the 3rd model. It's needed for teacher's forcing
        #encoder_inputs = Input(shape = (None,), name = "encoder_inputs") # same as the smiles_to_latent_model input
        #decoder_inputs = Input(shape = (None,), name = "decoder_inputs") 
        #building the full pipeline: smiles--> smiles
        x = self.smiles_to_latent_model(encoder_inputs)
        x = self.latent_to_states_model(x)
        x = [decoder_inputs] + x         # decoder inputs for teacher's forcing and x will be the reconstructed states
        x = self.states_to_smiles_model(x)

        #Full model
        self.model = Model(inputs = [encoder_inputs, decoder_inputs], outputs = [x], name = "Autoencoder")
        print(self.model.summary)



    def load_autoencoder_model(self, path):
        self.model.load_weights(path)
        self.build_sample_model()
        
    def fit_model(self,dataX, dataX2, dataY, epochs, batch_size, optimizer):

        self.epochs = epochs
        self.batch_size = batch_size
        #self.lr = lr

        if optimizer == 'adam':
            #self.optimizer = Adam(learning_rate = 0.005)
            self.optimizer = Adam(learning_rate = 0.001)
        elif optimizer == 'adam_clip':
            self.optimizer = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False, clipvalue = 3)


        ## Callbacks
        checkpoint_dir = self.path 
        #checkpoint_file = (checkpoint_dir + "%s--{epoch:02d}--{val_loss:.4f}--{lr:.7f}.hdf5" % model_name)
        checkpoint_file = (checkpoint_dir + "model--{epoch:02d}--{val_loss:.4f}.hdf5")
        checkpoint = ModelCheckpoint(checkpoint_file, monitor = "val_loss", mode = "min", save_best_only = True)
        
        #Reduces the learning rate by a factor of 2 when no improvement has been see in the validation set for 2 epochs
        reduce_lr = ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience=2, min_lr = 1e-6)
        
        #Early Stopping
        #early_stop = EarlyStopping(monitor = "val_loss", patience=5)
        
        callbacks_list = [checkpoint]#, reduce_lr]#, early_stop]


        self.model.compile(optimizer  = self.optimizer, loss = 'categorical_crossentropy')
        #self.model.compile(optimizer  = self.optimizer, loss =  'sparse_categorical_crossentropy')
       
        results = self.model.fit([dataX, dataX2], dataY, epochs = self.epochs, batch_size =self.batch_size, validation_split = 0.1, shuffle = True, verbose = 1, callbacks = callbacks_list)

        #last_epoch = early_stop.stopped_epoch

        fig, ax = plt.subplots()
        ax.plot(results.history['loss'], label = "Train")
        ax.plot(results.history['val_loss'], label = "Val")
        ax.legend()
        ax.set(xlabel='epochs', ylabel = 'loss')
        figure_path = self.path + "Loss_plot_"+str(dataX.shape[0])+".png"
        fig.savefig(figure_path)
        #plt.show()
        
        self.build_sample_model()
        
        #return last_epoch


    # def build_aux_model(self):

    #     decoder_inputs = Input(shape=self.input_shape, name = 'decoder_inputs')
    #     #x = Embedding(self.vocab_size, self.lstm_units//2)(decoder_inputs)
    #     #x = decoder_inputs
    #     inputs = [] #list that will have all the inputs to this model: decoder_inputs + reconstructed states (from 2nd model)

    #     inputs.append(decoder_inputs)

    #     x = decoder_inputs

    #     for dec_layer in range(self.numb_dec_layer): 
    #         name = "Decoded_state_h_" + str(dec_layer)
    #         state_h = Input(shape = [self.lstm_units], name = name)
    #         inputs.append(state_h)

    #         name = "Decoded_state_c_" + str(dec_layer)
    #         state_c = Input(shape = [self.lstm_units], name = name)
    #         inputs.append(state_c)

    #         #LSTM layer
    #         decoder_lstm = LSTM(self.lstm_units, return_sequences = True, name = "Decoder_LSTM_" + str(dec_layer))

    #         x = decoder_lstm(x, initial_state = [state_h, state_c])

    #         if self.batch_norm:
    #             x = BatchNormalization(momentum = self.batch_norm_momentum, name = "BN_decoder_"+str(dec_layer))(x)



    #         ################### if smt >0 Time Distributed


    #     #Dense layer that will return probabilities
    #     outputs = Dense(self.output_dim, activation = "softmax", name = "Decoder_Dense")(x)

    #     self.aux_model = Model(inputs = inputs, outputs = [outputs], name = "aux_model")

    #     #plot_model(self.states_to_smiles_model, to_file = 'states_to_model.png', show_shapes = True, show_layer_names = True)
    #     with open('aux_model.txt', 'w') as f:

    #         self.aux_model.summary(print_fn=lambda x: f.write(x + '\n'))






    def build_sample_model(self):
        

       # Get the configuration of the batch_model
        config = self.states_to_smiles_model.get_config()
       # new_config = config
        # Keep only the "Decoder_Inputs" as single input to the sample_model
        config["input_layers"] = [config["input_layers"][0]]

        # Find decoder states that are used as inputs in batch_model and remove them
        idx_list = []
        for idx, layer in enumerate(config["layers"]):

            if "Decoded_state_" in layer["name"]:
                idx_list.append(idx)

        # Pop the layer from the layer list
        # Revert indices to avoid re-arranging after deleting elements
        for idx in sorted(idx_list, reverse=True):
            config["layers"].pop(idx)

        # Remove inbound_nodes dependencies of remaining layers on deleted ones
        for layer in config["layers"]:
            idx_list = []

            try:
                for idx, inbound_node in enumerate(layer["inbound_nodes"][0]):
                    if "Decoded_state_" in inbound_node[0]:
                        idx_list.append(idx)
            # Catch the exception for first layer (Decoder_Inputs) that has empty list of inbound_nodes[0]
            except:
                pass

            # Pop the inbound_nodes from the list
            # Revert indices to avoid re-arranging
            for idx in sorted(idx_list, reverse=True):
                layer["inbound_nodes"][0].pop(idx)

        # Change the batch_shape of input layer
        config["layers"][0]["config"]["batch_input_shape"] = (
            1,
            1,
            self.output_dim,
        )

        # Finally, change the statefulness of the RNN layers
        for layer in config["layers"]:
            if "Decoder_LSTM_" in layer["name"]:
                layer["config"]["stateful"] = True
                # layer["config"]["return_sequences"] = True

        # Define the sample_model using the modified config file
        sample_model = Model.from_config(config)

        # Copy the trained weights from the trained batch_model to the untrained sample_model
        for layer in sample_model.layers:
            # Get weights from the batch_model
            weights = self.states_to_smiles_model.get_layer(layer.name).get_weights()
            # Set the weights to the sample_model
            sample_model.get_layer(layer.name).set_weights(weights)

        self.sample_model = sample_model
        return config
        
        
        
        
        
        
        
        #reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 10, min_lr=0.000001, verbose = 1, epsilon =1e-5)
        #Reduce learning rate when a metric has stopped improving.
        # factor: by which the learning rate will  be reduced: new_lr = lr*factor
        # patience: number of epochs with no improvement after which learning rate will be reduced.

    # def _____fit_model(self, dataX, dataY):

    #     results = self.model.fit([dataX, dataX], dataY, epochs = self.epochs, batch_size =self.batch_size, shuffle = True)
        
        
    #     fig, ax = plt.subplots()
    #     ax.plot(results.history['loss'])
    #     ax.set(xlabel='epochs', ylabel = 'loss')
    #     figure_path = "Loss_plot_"+str(dataX.shape[0])+".png"
    #     fig.savefig(figure_path)
    #     plt.show()
        
    #     self.smiles_to_latent_model = Model(self.encoder_inputs, self.neck_outputs)
    #     self.smiles_to_latent_model.save("smiles_to_latent" +str(dataX.shape[0])+".h5")
        
    #     latent_input = Input(shape=(self.latent_dim,))
    #     #reuse_layers
    #     state_h_decoded_2 =  self.decode_h(latent_input)
    #     state_c_decoded_2 =  self.decode_c(latent_input)
    #     self.latent_to_states_model = Model(latent_input, [state_h_decoded_2, state_c_decoded_2])
    #     self.latent_to_states_model.save("latent_to_states_" +str(dataX.shape[0])+".h5")
        
        
    #     #Decoder, we need to change it to stateful, and change the input shape
    #     inf_decoder_inputs = Input(batch_shape=(1, 1, self.input_shape[1]))
    #     inf_decoder_lstm = LSTM(self.lstm_units,
    #                         return_sequences=True,
    #                         stateful=True
    #                        )
    #     inf_decoder_outputs = inf_decoder_lstm(inf_decoder_inputs)
    #     inf_decoder_dense = Dense(self.output_dim, activation='softmax')
    #     inf_decoder_outputs = inf_decoder_dense(inf_decoder_outputs)
    #     self.sample_model = Model(inf_decoder_inputs, inf_decoder_outputs)
        
    #     #Transfer weights
    #     for i in range(1,3):
    #         self.sample_model.layers[i].set_weights(self.model.layers[i+6].get_weights())
    #     print(self.sample_model.summary())
    #     self.sample_model.save("sample_model_" +str(dataX.shape[0])+".h5")
        
    def latent_to_smiles(self, latent, vocab):  #sample
        '''
        Parameters
        ----------
        latent : TYPE latent representation of 1 smiles
            DESCRIPTION.
        vocab : TYPE Vocabulary object
            DESCRIPTION.

        Returns
        -------
        smiles : TYPE String
            DESCRIPTION. SMILES String predicted from the latent representation

        '''
        #predicts the c and h states from the latent representation
        states = self.latent_to_states_model.predict(latent)
        
        #updates the states in the sample model
        for dec_layer in range(self.numb_dec_layer): 
            self.sample_model.get_layer("Decoder_LSTM_"+ str(dec_layer)).reset_states(states = [states[2*dec_layer], states[2*dec_layer+1]])


        #self.sample_model.layers[1].reset_states(states = [states[0], states[1]])
        
        
        sample_vector = np.zeros(shape = (1,1,vocab.vocab_size))
        sample_vector[0,0,vocab.char_to_int["G"]] = 1  #input char
        smiles = ""
        for i in range(vocab.max_len):
            pred = self.sample_model.predict(sample_vector)
            idx = np.argmax(pred)
            char = vocab.int_to_char[idx]
            if char!= "A":
                smiles = smiles + char
                sample_vector = np.zeros((1,1,vocab.vocab_size))
                sample_vector[0,0, idx] = 1
            else:
                break
        smiles = vocab.replace_tokens_by_atoms(smiles)
        return smiles
        
def evaluate_reconstruction(real, predicted):
    
    assert len(real) == len(predicted)
    correct = 0
    for i in range(len(real)):
        if real[i] == predicted[i]:
            correct = correct+1
            
    #percentage of corrected reconstructed molecules
    return correct/len(real)*100    

def validity(smiles_list):
    '''
    Evaluates if the generated SMILES are valid using rdkit
    Parameters
    ----------
    smiles_list : TYPE
        DESCRIPTION. List of Smiles Strings

    Returns
    -------
    valid_smiles : TYPE List 
        DESCRIPTION. list of SMILES strings that were deamened valid
    perc_valid : TYPE
        DESCRIPTION. percentage of valid SMILES strings in the input data

    '''
    
    total = len(smiles_list)
    valid_smiles =[]
    count = 0
    for sm in smiles_list:
        if MolFromSmiles(sm) != None:
            valid_smiles.append(sm)
            count = count +1
    perc_valid = count/total*100
    
    return valid_smiles, perc_valid 

            
if __name__ == "__main__" :
    # testing/studying the Autoencoder's architecture
    

        
        
    #getting some data
    path = 'C:\\Users\\bjpsa\\Documents\\MIEB_Tese\\code\\AE\\'
    filename = 'ChEMBL_filtered'
    file = path + filename
    
    n =100000
    n2 = 1000 #testing data
    f_string = ''
    with open(file) as f:
        i = 0
        for line in f:
            #print(line)
            if len(line)<98:
                f_string = f_string+line
                i+=1
            if(i>=n+n2):
              break
    smiles = f_string.split('\n')[:-1]
    
   
    
    vocab = Vocabulary('Vocab.txt')
 
    
    vocab.update_vocab(smiles)
    tok = vocab.tokenize(smiles)
    
    tok_train = tok[n2:]
    tok_test = tok[0:n2]
    #one_hot = vocab.one_hot_encoder(tok) # ---> ready to be given as input to the LSTM
    X_train = vocab.one_hot_encoder(tok_train)
    Y_train = vocab.get_target(X_train)
    
    # X_test = vocab.one_hot_encoder(tok_test)
    # Y_test = vocab.get_target(X_test)
    
    latent_dim = 64
    lstm_units = 512
    epochs = 100
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = X_train.shape[1:] # = (max_len, vocab.size)
    output_dim = X_train.shape[-1] # = vocab.size
    auto = Autoencoder('', input_shape, latent_dim, lstm_units, output_dim, batch_norm, batch_norm_momentum, noise_std, numb_dec_layer)
    
    #Save model
    with open('Summary_autoencoder.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
        auto.model.summary(line_length=237, print_fn=lambda x: fh.write(x + '\n'))
    
   
    
    #auto.fit_model(X_train, Y_train, epochs, batch_size, 'adam',lr=0.001)
    
    
    print(auto.sample_model.summary())
    

    
