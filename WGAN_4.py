# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:06:08 2021

@author: bjpsa
"""
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Activation, Lambda, Layer, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import plot_model

from time import process_time, time
import numpy as np
import os
import csv
from matplotlib import pyplot as plt
from utils import *
import tqdm
import time

class WGANGP():
    def __init__(self, input_dim, critic_layers_units, critic_lr, critic_dropout, gp_weight, z_dim, generator_layers_units, generator_batch_norm_momentum, generator_lr, generator_dropout,batch_size, critic_optimizer, gen_optimizer):
        self.name = 'WGAN-GP'
        
        self.input_dim = input_dim
        self.critic_layers_units = critic_layers_units
        self.critic_nr_layers = len(critic_layers_units)
        self.critic_lr = critic_lr
        #self.critic_ativation = Activation(critic_activation)
        self.critic_dropout = critic_dropout
        
        self.gp_weight = gp_weight #gradient loss will be weighted by this factor
        
        self.z_dim = z_dim
        
        self.generator_layers_units = generator_layers_units
        self.generator_nr_layers = len(generator_layers_units)
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_lr = generator_lr
        self.generator_dropout =generator_dropout
        
        self.batch_size = batch_size
        self.optimizer_critic = self.get_optimizer(critic_optimizer, self.critic_lr)
        self.optimizer_generator = self.get_optimizer(gen_optimizer, self.generator_lr)
        
        self.epoch = 0
        self.d_losses_per_gen_iteration = []
        self.g_losses_per_gen_iteration = []
        self.gradient_penalty_loss = []
        self.critic_loss_real = []
        self.critic_loss_fake = []
        
        
        
        self.build_critic()
        self.build_generator()
        
    
    def get_optimizer(self, optimizer, lr):
        if optimizer == 'adam':
            opti = Adam(lr=lr, beta_1 = 0, beta_2 = 0.9) #valores defaul do paper original wgan-gp
        return opti
    
    def build_critic(self):
        
        critic_input = Input(shape = (self.input_dim,), name = 'critic_input')

        x = critic_input
        
        for i in range(self.critic_nr_layers-1):
            x = Dense(self.critic_layers_units[i], name = 'critic_layer_'+str(i))(x)
            x = LeakyReLU(alpha = 0.3)(x)   #default: alpha = 0.3 ;  paper: alpha = 0.2
            
            if self.critic_dropout > 0:
            	x = Dropout(self.critic_dropout)(x)
            
        
        x= Dense(1, activation = None, name = 'critic_layer_'+ str(i+1))(x)
        
        critic_output = x
        
        self.critic = Model(critic_input, critic_output, name = 'Critic')
        print(self.critic.summary())
        
    def build_generator(self):
        
        generator_input = Input(shape = (self.z_dim,), name = 'generator_input')
        
        x = generator_input
        
        for i in range(self.generator_nr_layers-1):
            x = Dense(self.generator_layers_units[i], name = 'generator_layer_'+str(i))(x)
            if self.generator_batch_norm_momentum:
                
                x  = BatchNormalization(momentum = self.generator_batch_norm_momentum, name = 'BN_'+str(i))(x)
            if self.generator_dropout >0:
            	x = Dropout(self.generator_dropout)(x)
            	
            x = LeakyReLU(alpha = 0.3)(x)   #default: alpha = 0.3 ;  paper: alpha = 0.2
        
        x = Dense(self.input_dim, activation = None, name = 'generator_layer_'+str(i+1))(x)
        
        generator_output = x
        
        self.generator = Model(generator_input, generator_output, name = 'Generator')
        print(self.generator.summary())
    
        
    def train_critic(self, x_train):
        
        # valid = np.ones((batch_size, 1), dtype = np.float32)
        # fake = -np.ones((batch_size, 1), dtype = np.float32)
        data = x_train
        noise = np.random.uniform(-1,1,(self.batch_size, self.z_dim))		#Drawing z vetors from a Normal distribution
        #dummy = np.zeros((batch_size, 1), dtype = np.float32) #for the GP
        #
        
        with tf.GradientTape() as critic_tape:
            self.critic.training = True
            #generating fake data
            generated_data = self.generator(noise)
            
            real_output = self.critic(data)
            fake_output = self.critic(generated_data)
            
            #wgan loss
            critic_loss = K.mean(fake_output)-K.mean(real_output)
            self.critic_loss_real.append(K.mean(real_output))
            self.critic_loss_fake.append(K.mean(fake_output))
            
            alpha = tf.random.uniform((self.batch_size,1))
            interpolated_samples = alpha*data +(1-alpha)*generated_data
            
            with tf.GradientTape() as t:
                t.watch(interpolated_samples)
                interpolated_samples_output = self.critic(interpolated_samples)
                
            gradients = t.gradient(interpolated_samples_output, [interpolated_samples])
            
            #computing the Euclidean/L2 Norm
            gradients_sqr = K.square(gradients)
            gradients_sqr_sum = K.sum(gradients_sqr, axis = np.arange(1, len(gradients_sqr.shape)))
            gradient_l2_norm = K.sqrt(gradients_sqr_sum)
            gradient_penalty = K.square(1-gradient_l2_norm) #returns the squared distance between L2 norm and 1
            #returns the mean over all the batch samples
            gp =  K.mean(gradient_penalty)
            
            self.gradient_penalty_loss.append(gp)
            #wgan-gp loss
            critic_loss = critic_loss +self.gp_weight*gp
            
        gradients_of_critic = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))
        
        #return K.sum(critic_loss)
        return critic_loss
    
    def train_generator(self):
        
        noise = np.random.normal(0,1,(self.batch_size, self.z_dim))
        
        with tf.GradientTape() as generator_tape:
            self.generator.training = True
            generated_data = self.generator(noise)
            
            fake_output = self.critic(generated_data)
            
            #wgan loss
            gen_loss = -K.mean(fake_output)
            #print('gen_loss', gen_loss, type(gen_loss))
            
        gradients_of_generator = generator_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.optimizer_generator.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        #return K.sum(gen_loss)
        return gen_loss
    
    def train(self, x_train, batch_size, epochs, run_folder, autoencoder, vocab, print_every_n_epochs, critic_loops = 5):
        self.n_critic = critic_loops
        self.autoencoder = autoencoder
        self.vocab = vocab
        self.data = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder = True).shuffle(buffer_size = x_train.shape[0])
        
        train_start = process_time()
        
        # gen_iter in range(self.generator_iterations, self.generator_iterations+ generator_iterations):
        self.g_loss_log = []
        self.critic_loss_log =[]
        
        self.critic_loss = []
        self.g_loss = []
            
        for epoch in range (self.epoch, self.epoch+epochs):
            critic_loss_per_batch = []
            g_loss_per_batch = []
            batches_done = 0
            

            for i, batch in enumerate(self.data):
                loss_d = self.train_critic(batch)
                
                critic_loss_per_batch.append(loss_d)
                
                # Train the Generator
                if i % self.n_critic == 0:
                    loss_g = self.train_generator()
                    
                    g_loss_per_batch.append(loss_g)
                    
                    batches_done = batches_done +  self.n_critic
                
                # Save information if it is the last batch ---> end of an epoch
                if i == len(self.data) -1:
                    
                    self.critic_loss_log.append([time.time(), epoch, np.mean(critic_loss_per_batch)])
                    self.g_loss_log.append([time.time(), epoch, np.mean(g_loss_per_batch)])
                    self.critic_loss.append(np.mean(critic_loss_per_batch))
                    self.g_loss.append(np.mean(g_loss_per_batch))		   
                    print( 'Epochs {}: D_loss = {}, G_loss = {}'.format(epoch, self.critic_loss_log[-1][2], self.g_loss_log[-1][2]))
            
                    if epoch % print_every_n_epochs == 0:# and epoch !=0:
                        print('!!!')
                        self.train_time = process_time()-train_start
                        #save 
                        self.save_model(run_folder)
                        self.critic.save_weights(os.path.join(run_folder, 'weights/critic_weights-%d.h5' % (epoch)))
                        self.critic.save_weights(os.path.join(run_folder, 'weights/critic_weights.h5'))
                        self.generator.save_weights(os.path.join(run_folder, 'weights/generator_weights-%d.h5' % (epoch)))
                        self.generator.save_weights(os.path.join(run_folder, 'weights/generator_weights.h5'))
                        self.sample_data(200, run_folder, save = True) 
                        self.plot_loss(run_folder)
                        train_start = process_time()
            self.epoch = self.epoch +1
        #self.plot_loss(run_folder)
    
    def train_feedbackGAN(self, x_train, x_train_smiles, batch_size, epochs, run_folder, autoencoder, vocab, predictor,n_to_generate, threshold, info, property_identifier, print_every_n_epochs, critic_loops = 5):
        self.n_critic = critic_loops
        self.autoencoder = autoencoder
        self.vocab = vocab
        self.epoch = 0
        #self.data = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder = True).shuffle(buffer_size = x_train.shape[0])
        
        train_start = process_time()
        
        # gen_iter in range(self.generator_iterations, self.generator_iterations+ generator_iterations):
        self.g_loss_log = []
        self.critic_loss_log =[]
        
        self.critic_loss = []
        self.g_loss = []
        
        #TIme measuring
        start=time.time()

        for epoch in range (self.epoch, self.epoch+epochs):
            
            #self.data = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder = True).shuffle(buffer_size = x_train.shape[0])
            self.data = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder = True)
            print('len(self.data)):', len(self.data))
            data_len = len(self.data)
            x_train_smiles = x_train_smiles[0:data_len*batch_size]
            
            critic_loss_per_batch = []
            g_loss_per_batch = []
            batches_done = 0
            train_epoch_start=process_time()

            for i, batch in enumerate(self.data):
                
                loss_d = self.train_critic(batch)
                
                critic_loss_per_batch.append(loss_d)
                
                # Train the Generator
                if i % self.n_critic == 0:
                    loss_g = self.train_generator()
                    
                    g_loss_per_batch.append(loss_g)
                    
                    batches_done = batches_done +  self.n_critic
                
                # Save information if it is the last batch ---> end of an epoch
                if i == len(self.data) -1:
                    
                    self.critic_loss_log.append([time.time(), epoch, np.mean(critic_loss_per_batch)])
                    self.g_loss_log.append([time.time(), epoch, np.mean(g_loss_per_batch)])
                    self.critic_loss.append(np.mean(critic_loss_per_batch))
                    self.g_loss.append(np.mean(g_loss_per_batch))		   
                    print( 'Epochs {}: D_loss = {}, G_loss = {}'.format(epoch, self.critic_loss_log[-1][2], self.g_loss_log[-1][2]))
            
                    
                    print('Saving weights....')
                    self.train_time = process_time()-train_start
                    #save 
                    self.save_model(run_folder)
                    self.critic.save_weights(os.path.join(run_folder, 'weights/critic_weights-%d.h5' % (epoch)))
                    self.generator.save_weights(os.path.join(run_folder, 'weights/generator_weights-%d.h5' % (epoch)))
                    self.plot_loss(run_folder)
                    train_start = process_time()
                        
                    #Sampling
                    gen_smiles, perc_valid, valid_generated_data = self.sample_valid_data(n_to_generate, run_folder, True)

                    #new_data, x_train_smiles, predictions = update_data_feedback_gan(x_train, x_train_smiles, gen_smiles, valid_generated_data, predictor, property_identifier, threshold, info)
                    new_data, x_train_smiles = update_data_feedback_gan_multi_obj(x_train, x_train_smiles, gen_smiles, valid_generated_data, predictor, property_identifier, threshold, info)
                    x_train = new_data
                    print('------------------------')
                    print(type(x_train_smiles))
                    print('x_train_smiles', len(x_train_smiles))
                    with open(os.path.join(run_folder, "new_data_epoch_%d.csv" % (self.epoch)), 'w')as f:
                        writer = csv.writer(f)
                        for i in range(new_data.shape[0]):
                            writer.writerow(new_data[i])
                    
                    #save as csv
                    #with open(os.path.join(run_folder, "feedbackGAN_results.csv"), 'a') as f:
                     #   writer = csv.writer(f)
                     #   writer.writerow([self.epoch, np.max(predictions), np.mean(predictions), np.min(predictions), perc_valid])
            print(epoch)        
            self.epoch = self.epoch +1
            end_epoch_time=process_time()
            my_file = open("training_epoch_time.txt", "a")
            my_string="Time spent {}, iteration {} \n".format(end_epoch_time-train_epoch_start,i)
            my_file.write(my_string)
        #self.plot_loss(run_folder)
    
    
    def sample_data(self, n, run_folder, save):
        print('sampling data...')
        noise = np.random.uniform(-1,1,(n, self.z_dim))       #generates noise vectors
        generated_data = self.generator.predict(noise)      #generates fake data
        generated_smiles = []
        for i in range(generated_data.shape[0]):            #transforms fake data into SMILES
            sml = self.autoencoder.latent_to_smiles(generated_data[i:i+1], self.vocab)
            generated_smiles.append(sml)
        
        valid_smiles, perc_valid, _ = validity(generated_smiles)
        if save == True: 
            #with open(os.path.join(run_folder, "generated_data/samples_epoch_%d_val_%0.2f.csv" % (self.epoch, valid)), 'w') as f:
            with open(os.path.join(run_folder, "samples_epoch_%d_val_%0.2f.csv" % (self.epoch, perc_valid)), 'w') as f:
    	        writer = csv.writer(f)
    	        for i in range(len(generated_smiles)):
    	        	writer.writerow(generated_smiles[i])
            #row = [self.epoch, valid, secondsToStr(self.train_time)]
            #with open(os.path.join(run_folder, "generated_data/results.csv"), 'a') as f:
            #    writer = csv.writer(f)
            #    writer.writerow(row)
                
        return valid_smiles

    def sample_valid_data(self, n, run_folder, save):

        print('sampling data...')
        aux = n
        valid_smiles  = []
        valid_generated_data = []
        perc_valid = 0
        train_epoch_start=process_time()
        total_generated=0
        
        #while (aux <n):
        temp=aux
        while (temp >0):
            #if aux == n &&:
            if self.epoch % 50 == 0 and self.epoch !=0: 
                noise = np.random.uniform(-1,1,(1000, self.z_dim))       #generates noise vectors
                generated_data = self.generator.predict(noise)      #generates fake data
                print(generated_data)
                print(generated_data.shape)
                generated_smiles = []
                for i in range(generated_data.shape[0]):            #transforms fake data into SMILES
                    sml = self.autoencoder.latent_to_smiles(generated_data[i:i+1], self.vocab)
                    generated_smiles.append(sml)
                valid_smiles_aux, perc_valid, idx = validity(generated_smiles)
                
                for ind in idx:
                    valid_generated_data.append(generated_data[ind:ind+1])
                       
            else:
                noise = np.random.uniform(-1,1,(temp, self.z_dim))       #generates noise vectors
                generated_data = self.generator.predict(noise)      #generates fake data
                generated_smiles = []
                for i in range(generated_data.shape[0]):            #transforms fake data into SMILES
                    sml = self.autoencoder.latent_to_smiles(generated_data[i:i+1], self.vocab)
                    generated_smiles.append(sml)
        
                valid_smiles_aux, perc_valid, idx = validity(generated_smiles)
                total_generated+=len(generated_smiles)
                for ind in idx:
                    valid_generated_data.append(generated_data[ind:ind+1])
            
            valid_smiles = valid_smiles + valid_smiles_aux
            
            
            if len(valid_smiles)>n:
                valid_smiles = valid_smiles[0:n]
                valid_generated_data = valid_generated_data[0:n]
            #aux = len(valid_smiles)
            temp = aux-len(valid_smiles)
            
        
        if (save == True) and (self.epoch % 50 ==0): 
            
            perc_valid=(n/float(total_generated))*100.0
            

            with open(os.path.join(run_folder, "unbiased_valid_1000_samples_epoch_%d_val_%0.2f.csv" % (self.epoch, perc_valid)), 'w') as f:
                writer = csv.writer(f)
                for i in range(len(valid_smiles)):
                    writer.writerow(valid_smiles[i])

        print("Valid generated data")
        print(valid_smiles)
        end_epoch_time=process_time()
        my_file = open("sampling_data.txt", "a")
        my_string="Time spent {}, iteration {} \n".format(end_epoch_time-train_epoch_start,n)
        my_file.write(my_string)

        return valid_smiles, perc_valid, np.array(valid_generated_data)
        
    def save_model(self, run_folder):
     	#self.model.save(os.path.join(run_folder, 'model.h5'))
     	self.critic.save(os.path.join(run_folder, 'critic.h5'))
     	self.generator.save(os.path.join(run_folder, 'generator.h5'))

    def load_weights(self, filepath_critic, filepath_generator):
        self.critic.load_weights(filepath_critic)
        self.generator.load_weights(filepath_generator)
        
    def plot_loss(self, run_folder):
        fig, ax = plt.subplots()
        ax.plot(self.g_loss, label = "G_loss")
        ax.plot(self.critic_loss, label = "D_loss")
        ax.legend()
        ax.set(xlabel='Epoch', ylabel = 'loss')
        figure_path = os.path.join(run_folder, 'viz/Loss_plot_gen_iterations_%d.png'% (self.epoch))
        #figure_path = run_folder + "Loss_plot.png"
        fig.savefig(figure_path)
        plt.close()
    def plot_model(self, run_folder):
    	plot_model(self.critic, to_file=os.path.join(run_folder, 'viz/critic.png', show_shapes = True, show_layer_names = True))
    	plot_model(self.generator, to_file=os.path.join(run_folder, 'viz/generator.png', show_shapes = True, show_layer_names = True))
        
        
            
            
