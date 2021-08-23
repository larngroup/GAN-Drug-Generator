# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:02:00 2021

@author: bjpsa
"""


import json
from bunch import Bunch
import time
import os
import csv
import numpy as np
import random
import math
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from functools import reduce
import pandas as pd
import seaborn as sb
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AllChem, QED, Descriptors
from rdkit import DataStructs
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from sascorer_calculator import SAscore


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
    idx = []
    count = 0
    for i,sm in enumerate(smiles_list):
        if MolFromSmiles(sm) != None:
            valid_smiles.append(sm)
            idx.append(i)
            count = count +1
    perc_valid = count/total*100
    
    return valid_smiles, perc_valid, idx

def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t*1000,), 1000, 60, 60])
            
    
#### from Tiago

def uniqueness(smiles_list):
    
    valid_smiles, _ , _= validity(smiles_list)

    unique_smiles = list(set(valid_smiles))
    
    return (len(unique_smiles)/len(valid_smiles))*100

def diversity(smiles_A,smiles_B = None):
# # If you want to compute internal similarity just put the 
# # filename_a and the filename_b as 'None'. If you want to compare two sets, 
# # write its names properly and it will be computed the Tanimoto distance.
# # Note that it is the Tanimoto distance, not Tanimoto similarity. 
 
    td = 0
    #print(smiles_A)
    #print(smiles_B)
    fps_A = []
    for i, row in enumerate(smiles_A):
        try:
            mol = MolFromSmiles(row)
            fps_A.append(AllChem.GetMorganFingerprint(mol, 3))
        except:
            print('ERROR: Invalid SMILES!')
            
        
    
    if smiles_B == None:
        for ii in range(len(fps_A)):
            for xx in range(len(fps_A)):
                ts = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
                td += ts          
        
        if len(fps_A) == 0:
            td = None
        else:
            td = td/len(fps_A)**2
    else:
        fps_B = []
        for j, row in enumerate(smiles_B):
            try:
                mol = MolFromSmiles(row)
                fps_B.append(AllChem.GetMorganFingerprint(mol, 3))
            except:
                print('ERROR: Invalid SMILES!') 
        
        
        for jj in range(len(fps_A)):
            for xx in range(len(fps_B)):
                ts = 1 - DataStructs.TanimotoSimilarity(fps_A[jj], fps_B[xx]) 
                td += ts
        
        if (len(fps_A) == 0 or len(fps_B) == 0):
            td = None
        else:   
            td = td / (len(fps_A)*len(fps_B))
    print("Tanimoto distance: " + str(td))  
    return td


def external_diversity(file_A,file_B):

    td = 0
    file_A = [file_A]
    fps_A = []
    for i, row in enumerate(file_A):
        try:
            mol = MolFromSmiles(row)
            fps_A.append(AllChem.GetMorganFingerprint(mol, 6))
        except:
            print('ERROR: Invalid SMILES!')
            
        
    
    if file_B == None:
        for ii in range(len(fps_A)):
            for xx in range(len(fps_A)):
                ts = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
                td += ts          
      
        td = td/len(fps_A)**2
    else:
        fps_B = []
        for j, row in enumerate(file_B):
            try:
                mol = MolFromSmiles(row)
                fps_B.append(AllChem.GetMorganFingerprint(mol, 6))
            except:
                print('ERROR: Invalid SMILES!') 
        
        
        for jj in range(len(fps_A)):
            for xx in range(len(fps_B)):
                ts = 1 - DataStructs.TanimotoSimilarity(fps_A[jj], fps_B[xx]) 
                td += ts
        
        td = td / (len(fps_A)*len(fps_B))
    print("Tanimoto distance: " + str(td))  
    return td

def diversity_3(smiles_list):
    """
    Function that takes as input a list containing SMILES strings to compute
    its internal diversity
    Parameters
    ----------
    smiles_list: List with valid SMILES strings
    Returns
    -------
    This function returns the internal diversity of the list given as input, 
    based on the computation Tanimoto similarity
    """
    td = 0
    
    fps_A = []
    for i, row in enumerate(smiles_list):
        try:
            mol = MolFromSmiles(row)
            fps_A.append(AllChem.GetMorganFingerprint(mol, 6))
        except:
            print('ERROR: Invalid SMILES!')
            
        
    for ii in range(len(fps_A)):
        for xx in range(len(fps_A)):
            tdi = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
            td += tdi          
      
    td = td/len(fps_A)**2

    return td

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def rmse(y_true, y_pred):
    """
    This function implements the root mean squared error measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    Returns
    -------
    Returns the rmse metric to evaluate regressions
    """

    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def mse(y_true, y_pred):
    """
    This function implements the mean squared error measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    Returns
    -------
    Returns the mse metric to evaluate regressions
    """
    return K.mean(K.square(y_pred - y_true), axis=-1)

def r_square(y_true, y_pred):
    """
    This function implements the coefficient of determination (R^2) measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    Returns
    -------
    Returns the R^2 metric to evaluate regressions
    """

    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def ccc(y_true,y_pred):
    """
    This function implements the concordance correlation coefficient (ccc)
    ----------
    y_true: True label   
    y_pred: Model predictions 
    Returns
    -------
    Returns the ccc measure that is more suitable to evaluate regressions.
    """
    num = 2*K.sum((y_true-K.mean(y_true))*(y_pred-K.mean(y_pred)))
    den = K.sum(K.square(y_true-K.mean(y_true))) + K.sum(K.square(y_pred-K.mean(y_pred))) + K.int_shape(y_pred)[-1]*K.square(K.mean(y_true)-K.mean(y_pred))
    return num/den

def load_config(config_file,property_identifier):
    """
    This function loads the configuration file in .json format. Besides, it 
    creates the directory of this experiment to save the created models
    ----------
    config_file: name of the configuration file;
    property_identifier: string that indicates the property we will use;
    Returns
    -------
    This function returns the configuration file.
    """
    print("Loading configuration file...")
    
    with open(config_file, 'r') as config_file:
        config_dict = json.load(config_file)
        config = Bunch(config_dict)
        exp_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        config.checkpoint_dir = os.path.join('experiments',property_identifier + '-' + exp_time+'\\', config.exp_name, 'checkpoints\\')
        #config.output = config.output + exp_time
    print("Configuration file loaded successfully!")
    return config;

def directories(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print('Creating directories error: {}'.format(err))
        exit(-1)
        
        
        
def transform_to_array(X):
    #print(X)
    max_l = 0
    for i in X:
        length = len(i)
        if length > max_l:
            max_l = length
    aux_arr = np.zeros(shape = (X.shape[0], max_l))
    for k in range(X.shape[0]):
        aux_arr[k] = np.asarray(X[k])
    return aux_arr
        
def reading_csv(config,property_identifier):
    """
    This function loads the SMILES strings and the respective labels of the 
    specified property by the identifier.
    ----------
    config: configuration file
    property_identifier: Identifier of the property we will use. It could be 
    (jak2,logP or kor)
    
    Returns
    -------
    smiles, labels: Lists with the loaded data. We only select the SMILES with
    length under a certain threshold defined in the configuration file. Also, 
    remove the duplicates in the original dataset.
    """
    if property_identifier == 'bbb':
        filepath = config.datapath_jak2
        idx_smiles = 0
        idx_labels = 1
    elif property_identifier == 'a2d':
        filepath = config.datapath_a2d
        idx_smiles = 0
        idx_labels = 1
    elif property_identifier == 'kor':
        filepath = 'data/data_clean_kop.csv'#+config.datapath_kor
        idx_smiles = 0
        idx_labels = 1   
    elif property_identifier == 'jak2':
        filepath = 'data/jak2_data.csv'
        idx_smiles = 0
        idx_labels = 1
        
    raw_smiles = []
    raw_labels = []
    
    with open(filepath, 'r') as csvFile:
        reader = csv.reader(csvFile)
        
        it = iter(reader)
        next(it, None)  # skip first item.   
        permeable = 0
        for row in it:
            try:
                if "[S@@H0]" in row[idx_smiles] or "[n+]" in row[idx_smiles] or "[o+]" in row[idx_smiles] or "[c@@]" in row[idx_smiles]:
                    print("-->",row[idx_smiles])
                elif permeable < 1249 or float(row[idx_labels]) == 0:
                    raw_smiles.append(row[idx_smiles])
                    raw_labels.append(float(row[idx_labels]))
                    if float(row[idx_labels]) == 1:
                        permeable = permeable + 1

            except:
                pass
    
    smiles = []
    labels = []
#    and raw_smiles[i] not in smiles
    #and 'L' not in raw_smiles[i] and 'Cl' not in raw_smiles[i] and 'Br' not in raw_smiles[i]
    for i in range(len(raw_smiles)):
        if len(raw_smiles[i]) <= config.smile_len_threshold  and 'a' not in raw_smiles[i] and 'Z' not in raw_smiles[i] and 'K' not in raw_smiles[i]:
            smiles.append(raw_smiles[i])
            labels.append(raw_labels[i])
            
    return smiles, labels

def data_division(config,smiles_int,labels,cross_validation,model_type,descriptor):
    """
    This function divides data in two or three sets. If we are performing 
    grid_search we divide between training, validation and testing sets. On 
    the other hand, if we are doing cross-validation, we just divide between 
    train/validation and test sets because the train/validation set will be then
    divided during CV.
    ----------
    config: configuration file;
    smiles_int: List with SMILES strings set;
    labels: List with label property set;
    cross_validation: Boolean indicating if we are dividing data to perform 
                      cross_validation or not;
    model_type: String indicating the type of model (dnn, SVR, KNN or RF)
    descriptor: String indicating the descriptor (ECFP or SMILES)
    Returns
    -------
    data: List with the sets of the splitted data.
    """ 
    data = []
    
    idx_test = np.array(random.sample(range(0, len(smiles_int)), math.floor(config.percentage_test*len(smiles_int))))
    train_val_set = np.delete(smiles_int,idx_test,0)
    train_val_labels = np.delete(labels,idx_test)
    
    test_set = np.array(smiles_int)[idx_test.astype(int)]
    labels = np.array(labels)
    test_labels = labels[idx_test]
    
    if cross_validation:
        data.append(train_val_set)
        data.append(train_val_labels)
        data.append(test_set)
        data.append(test_labels)
    else:
        idx_val = np.array(random.sample(range(0, len(train_val_set)), math.floor(config.percentage_test*len(train_val_set))))
        train_set = np.delete(train_val_set,idx_val,0)
        train_labels = np.delete(train_val_labels,idx_val)
        val_set = train_val_set[idx_val]
        train_val_labels = np.array(train_val_labels)
        val_labels = train_val_labels[idx_val]
        
        data.append(train_set)
        data.append(train_labels)
        data.append(test_set)
        data.append(test_labels)
        data.append(val_set)
        data.append(val_labels)
    
    return data

def cv_split(data,config):
    """
    This function performs the data spliting into 5 consecutive folds. Each 
    fold is then used once as a test set while the 4 remaining folds 
    form the training set.
    ----------
    config: configuration file;
    data: List with the list of SMILES strings set and a list with the label;
    Returns
    -------
    data: object that contains the indexes for training and testing for the 5 
          folds
    """
    train_val_smiles = data[0]
    train_val_labels = data[1]
    cross_validation_split = KFold(n_splits=config.n_splits, shuffle=True)
    data_cv = list(cross_validation_split.split(train_val_smiles, train_val_labels))
    return data_cv

def normalize(data):
    """
    This function implements the percentile normalization step (to avoid the 
    interference of outliers).
    ----------
    data: List of label lists. It contains the y_train, y_test, and y_val (validation)
    Returns
    -------
    Returns z_train, z_test, z_val (normalized targets) and data (values to 
    perform the denormalization step). 
    """
    data_aux = np.zeros(2)
    y_train = data[1]
    y_test = data[3]
    y_val = data[5]
#    m_train = np.mean(y_train)
#    sd_train = np.std(y_train)
#    m_test = np.mean(y_test)
#    sd_test = np.std(y_test)
#    
#    z_train = (y_train - m_train) / sd_train
#    z_test = (y_test - m_test) / sd_test
#    
#    max_train = np.max(y_train)
#    min_train = np.min(y_train)
#    max_val = np.max(y_val)
#    min_val = np.min(y_val)
#    max_test = np.max(y_test)
#    min_test = np.min(y_test)
#    
    q1_train = np.percentile(y_train, 5)
    q3_train = np.percentile(y_train, 90)
#    
    q1_test = np.percentile(y_test, 5)
    q3_test = np.percentile(y_test, 90)
    
    q1_val = np.percentile(y_val, 5)
    q3_val = np.percentile(y_val, 90)

#    z_train = (y_train - min_train) / (max_train - min_train)
#    z_test = (y_test - min_test) / (max_test - min_test)
    
#    data[1] = (y_train - q1_train) / (q3_train - q1_train)
#    data[3]  = (y_test - q1_test) / (q3_test - q1_test)
#    data[5]  = (y_val - q1_val) / (q3_val - q1_val)
    data[1] = (y_train - q1_train) / (q3_train - q1_train)
    data[3]  = (y_test - q1_test) / (q3_test- q1_test)
    data[5]  = (y_val - q1_val) / (q3_val - q1_val)
    
    
    data_aux[1] = q1_train
    data_aux[0] = q3_train
#    data[2] = m_train
#    data[3] = sd_test
   
    return data,data_aux

def denormalization(predictions,data):
    """
    This function implements the denormalization step.
    ----------
    predictions: Output from the model
    data: q3 and q1 values to perform the denormalization
    Returns
    -------
    Returns the denormalized predictions.
    """
    for l in range(len(predictions)):
        
        max_train = data[l][0]
        min_train = data[l][1]
#        m_train = data[l][2]
#        sd_train = data[l][3]
       
        for c in range(len(predictions[0])):
            predictions[l,c] = (max_train - min_train) * predictions[l,c] + min_train
#            predictions[l,c] = predictions[l,c] * sd_train + m_train
  
    return predictions

def denormalization_with_labels(predictions,labels):   
    """
    This function performs the denormalization of the Predictor output.
    ----------
    predictions: list with the desired property predictions.
    labels: list with the labels of the desired property training data.
    
    Returns
    -------
    predictions: Returns the denormalized predictions.
    """
    
    for l in range(len(predictions)):
        
        q1 = np.percentile(labels,5)
        q3 = np.percentile(labels,95)
        for c in range(len(predictions[0])):
            predictions[l,c] = (q3 - q1) * predictions[l,c] + q1
#            predictions[l,c] = predictions[l,c] * sd_train + m_train
          
    return predictions

def regression_plot(y_true,y_pred):
    """
    Function that graphs a scatter plot and the respective regression line to 
    evaluate the QSAR models.
    Parameters
    ----------
    y_true: True values from the label
    y_pred: Predictions obtained from the model
    Returns
    -------
    This function returns a scatter plot.
    """
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], 'k--', lw=4)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    plt.show()
    fig.savefig('regression.png')


def get_reward(predictor, smile, memory_smiles, property_identifier):
    """
    This function takes the predictor model and the SMILES string to return 
    a numerical reward for the specified property
    ----------
    predictor: object of the predictive model that accepts a trajectory
        and returns a numerical prediction of desired property for the given 
        trajectory
    smile: generated molecule SMILES string
    property_identifier: String that indicates the property to optimize
    Returns
    -------
    Outputs the reward value for the predicted property of the input SMILES 
    """
    print('getReward')
    list_ss = [smile]
    
    if property_identifier == 'kor':
        
        pred = predictor.predict(list_ss)
        reward = np.exp(pred/4-1)
        
    diversity = 1
    if len(memory_smiles)>20:
        diversity = external_diversity(smile, memory_smiles)
    
    if diversity <0.75:
        rew_div = 0.9
        print("\nRepetition")
    elif diversity >0.9:
        rew_div = 1
    else:
        rew_div = 1
    print('output getreward:', reward*rew_div)
    return (reward*rew_div)
    

def moving_average(previous_values, new_value, ma_window_size=10): 
    """
    This function performs a simple moving average between the previous 9 and the
    last one reward value obtained.
    ----------
    previous_values: list with previous values 
    new_value: new value to append, to compute the average with the last ten 
               elements
    
    Returns
    -------
    Outputs the average of the last 10 elements 
    """
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def plot_training_progress(training_rewards,training_losses):
    """
    This function plots the progress of the training performance
    ----------
    training_rewards: list with previous reward values
    training_losses: list with previous loss values
    """
    plt.plot(training_rewards)
    plt.xlabel('Training iterations')
    plt.ylabel('Average rewards')
    plt.show()
    plt.plot(training_losses)
    plt.xlabel('Training iterations')
    plt.ylabel('Average losses')
    plt.show()
    
def sample_data_from_generator(generator, autoencoder, vocab, z_dim, n, run_folder, save):
    print('sampling data...')
    noise = np.random.uniform(-1,1,(n, z_dim))       #generates noise vectors
    generated_data = generator.predict(noise)      #generates fake data
    generated_smiles = []
    for i in range(generated_data.shape[0]):            #transforms fake data into SMILES
        sml = autoencoder.latent_to_smiles(generated_data[i:i+1], vocab)
        generated_smiles.append(sml)
    
    valid_smiles, perc_valid = validity(generated_smiles)
#     if save == True: 
#         #with open(os.path.join(run_folder, "generated_data/samples_epoch_%d_val_%0.2f.csv" % (self.epoch, valid)), 'w') as f:
#         with open(os.path.join(run_folder, "samples_epoch_%d_val_%0.2f.csv" % (self.epoch, perc_valid)), 'w') as f:
# 	        writer = csv.writer(f)
# 	        for i in range(len(generated_smiles)):
# 	        	writer.writerow(generated_smiles[i])
#         #row = [self.epoch, valid, secondsToStr(self.train_time)]
#         #with open(os.path.join(run_folder, "generated_data/results.csv"), 'a') as f:
#         #    writer = csv.writer(f)
#         #    writer.writerow(row)
            
    return valid_smiles

def qed_calculator(mols):
    """
    Function that takes as input a list of SMILES to predict its qed value
    Parameters
    ----------
    mols: list of molecules
    Returns
    -------
    This function returns a list of qed values 
    """
    qed_values = []
    for mol in mols:
        try:
            q = QED.qed(mol)
            qed_values.append(q)
        except: 
            print('unable to compute QED')
            qed_values.append('NaN')
            pass

        
    return qed_values

def logPcalculator(list_smiles):
    
    predictions = []
    for smile in list_smiles:
        try:
            mol = MolFromSmiles(smile)
            logP = Descriptors.MolLogP(mol)
            predictions.append(logP)
        except:
            print('Invalid')
            
    return predictions

def MW_calculator(mols):
    """
    Function that takes as input a list of Molecules and predicts the Molecular Wr«eight

    Returns
    -------
    This function returns a list of molecular weights

    """
    mw_values = []
    for mol in mols:
        mw = Descriptors.MolWt(mol)
        mw_values.append(mw)
        
    return mw_values
    

def evaluate_property(predictor, smiles, property_identifier):
    
    if property_identifier == 'qed':
        qed_values = qed_calculator(smiles2mol(smiles))
        return qed_values
    elif property_identifier == 'kor':
        kor_values, og_idx = predictor.predict(smiles)
        return kor_values, og_idx
    elif property_identifier == 'logP':
        logP_values = logPcalculator(smiles)
        return logP_values
    elif property_identifier == 'sascore':
        sascore = SAscore(smiles2mol(smiles))
        return sascore
    elif property_identifier == 'mw':
        mw_values = MW_calculator(smiles2mol(smiles))
        return mw_values
    
def smiles2mol(smiles_list):
    """
    Function that converts a list of SMILES strings to a list of RDKit molecules 
    Parameters
    ----------
    smiles: List of SMILES strings
    ----------
    Returns list of molecules objects 
    """
    mol_list = []
    if isinstance(smiles_list,str):
        mol = Chem.MolFromSmiles(smiles_list, sanitize=True)
        mol_list.append(mol)
    else:
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            mol_list.append(mol)
    return mol_list

def update_data_feedback_gan(previous_data, previous_smiles, gen_smiles, valid_generated_data, predictor, property_identifier, threshold, info):
    # previous_data --> numpy array with latent vectors
    # previous_smiles --> list with SMILES strings
    # gen_smiles --> list with generated valid smiles strings
    # valid_generated_data --> numpy array with generated latent vectors
    print('previous_Data', previous_data.shape)
    print('previous smiles', len(previous_smiles))
    # evaluate property using the Predictor
    eval_gen_smiles, og_idx = evaluate_property(predictor,gen_smiles, property_identifier)
    gen_smiles_2 = [gen_smiles[i] for i in og_idx] 
    gen_smiles = gen_smiles_2
    valid_generated_data_2 = [valid_generated_data[i] for i in og_idx]
    valid_generated_data = valid_generated_data_2
    eval_previous_smiles, _ = evaluate_property(predictor, previous_smiles, property_identifier)
    
    
    #sort previous smiles according to property
    
    eval_previous_smiles_arr = np.array(eval_previous_smiles)
    previous_smiles_arr = np.array(previous_smiles)
    inds = eval_previous_smiles_arr.argsort()
    
    print(inds.shape)
    print('previous_data', previous_data.shape)
    print('gen_smiles', len(gen_smiles))
    
    
    sorted_previous_data = previous_data[inds]
    sorted_previous_smiles = list(previous_smiles_arr[inds])
    
    #sort new smiles
    eval_gen_smiles_arr = np.array(eval_gen_smiles)
    gen_smiles_arr = np.array(gen_smiles)
    inds = eval_gen_smiles_arr.argsort()
    
    sorted_gen_data = np.squeeze(np.array(valid_generated_data))[inds]
    sorted_gen_smiles = list(gen_smiles_arr[inds])
    
    initial_n = previous_data.shape[0]
    
    new_smiles =[]
    #new_data = previous_data
    idx_save = []    

    if info == 'max':
        for idx, j in enumerate(eval_gen_smiles):
            if sorted_gen_smiles[idx] not in sorted_previous_smiles:
                idx_save.append(idx)
                new_smiles.append(sorted_gen_smiles[idx]) # SMILES strings
        
        sorted_new_data_ = [sorted_gen_data[i] for i in idx_save]  

        n = 20
        if len(new_smiles) <20:
            n = len(new_smiles) 

        output = np.vstack((sorted_previous_data[n:sorted_previous_data.shape[0],:], np.squeeze(np.array(sorted_new_data_[len(sorted_new_data_)-n:len(sorted_new_data_)]))))
        
        output_smiles = sorted_previous_smiles[n:len(sorted_previous_smiles)]+ new_smiles[len(new_smiles)-n:len(new_smiles)]
        
        eval_smiles, og_idx = evaluate_property(predictor,new_smiles[len(new_smiles)-n:len(new_smiles)], property_identifier)

    elif info == 'min':
         print('min')
        for idx, j in enumerate(eval_gen_smiles):
            if sorted_gen_smiles[idx] not in sorted_previous_smiles:
                idx_save.append(idx)
                new_smiles.append(sorted_gen_smiles[idx]) # SMILES strings
        #sorted_new_data_ = [sorted_new_data[i] for i in idx_save]      
        sorted_new_data_ = [sorted_gen_data[i] for i in idx_save]   
        #n = len(new_smiles) 
        n = 20
        if len(new_smiles) <20:
            n = len(new_smiles) 
            
            
        output = np.vstack((sorted_previous_data[0:sorted_previous_data.shape[0]-n,:], np.squeeze(np.array(sorted_new_data_[0:n]))))
        
        output_smiles = sorted_previous_smiles[0:len(sorted_previous_smiles)-n]+ new_smiles[0:n]  
        eval_smiles, og_idx = evaluate_property(predictor,new_smiles[0:n], property_identifier)  
        print('new predictions', eval_smiles)

    print('novas moleculas:', n)
    print('output_data', output.shape)
    print('output_smiles', len(output_smiles))
    predictions = eval_smiles
    print(type(output_smiles))
    
    assert len(output_smiles) == output.shape[0]
    
    return output, output_smiles, predictions

def plot_hist_both(prediction_unb,prediction_b, property_identifier):
    """
    Function that plots the predictions's distribution of the generated SMILES 
    strings, obtained by the unbiased and biased generators.
    Parameters
    ----------
    prediction_unb: list with the desired property predictions of unbiased 
                    generator.
    prediction_b: list with the desired property predictions of biased 
                    generator.
    property_identifier: String identifying the property 
    Returns
    ----------
    This functions returns the difference between the averages of the predicted
    properties
    """
    prediction_unb = np.array(prediction_unb)
    prediction_b = np.array(prediction_b)
    
    legend_unb = ''
    legend_b = '' 
    label = ''
    plot_title = ''
    
    
    if property_identifier == 'jak2' or property_identifier == "kor":
        legend_unb = 'Unbiased pIC50 values'
        legend_b = 'Biased pIC50 values'
        print("Max of pIC50: (UNB,B)", np.max(prediction_unb),np.max(prediction_b))
        print("Mean of pIC50: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Min of pIC50: (UNB,B)", np.min(prediction_unb),np.min(prediction_b))
        print("Std of pIC50: (UNB, B)", np.std(prediction_unb), np.std(prediction_b))
    
        label = 'Predicted pIC50'
        #plot_title = 'Distribution of predicted pIC50 for generated molecules'
        plot_title = 'Distribution of predicted pIC50'
        
    elif property_identifier == "sas":
        legend_unb = 'Unbiased SA score values'
        legend_b = 'Biased SA score values'
        print("Max of SA score: (UNB,B)", np.max(prediction_unb),np.max(prediction_b))
        print("Mean of SA score: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Min of SA score: (UNB,B)", np.min(prediction_unb),np.min(prediction_b))
    
        label = 'Predicted SA score'
        plot_title = 'Distribution of SA score values for generated molecules'  
    elif property_identifier == "qed":
        legend_unb = 'Unbiased QED values'
        legend_b = 'Biased QED values'
        print("Max of QED: (UNB,B)", np.max(prediction_unb),np.max(prediction_b))
        print("Mean of QED: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Min of QED: (UNB,B)", np.min(prediction_unb),np.min(prediction_b))
    
        label = 'Predicted QED'
        plot_title = 'Distribution of QED values for generated molecules'  
    elif property_identifier == 'logP':
        legend_unb = 'Unbiased logP values'
        legend_b = 'Biased logP values'
        
        percentage_in_threshold_unb = np.sum((prediction_unb >= 0.0) & 
                                            (prediction_unb <= 5.0))/len(prediction_unb)
        percentage_in_threshold_b = np.sum((prediction_b >= 0.0) & 
                                 (prediction_b <= 5.0))/len(prediction_b)
        print("% of predictions within drug-like region (UNB,B):", 
          percentage_in_threshold_unb,percentage_in_threshold_b)
        print("Average of log_P: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Median of log_P: (UNB,B)", np.median(prediction_unb),np.median(prediction_b))
    
        label = 'Predicted logP'
        plot_title = 'Distribution of predicted LogP for generated molecules'
        plt.axvline(x=0.0)
        plt.axvline(x=5.0)
        
    v1 = pd.Series(prediction_unb, name=legend_unb)
    v2 = pd.Series(prediction_b, name=legend_b)
   
    sb.set_palette("YlGnBu_r")
    # ax = sb.kdeplot(v1, shade=True,color='b')
    # sb.kdeplot(v2, shade=True,color='g')
    ax = sb.kdeplot(v1, shade=True)
    sb.kdeplot(v2, shade=True)

    ax.set(xlabel=label,ylabel = 'Density', 
           title=plot_title)
    plt.legend(loc='upper right', labels=[legend_unb, legend_b])
    #plt.show()
    plt.savefig('Plot feedbackGAN')
    plt.show()
    plt.close()
    
    return np.mean(prediction_b) - np.mean(prediction_unb)


def plot_multiple_histograms(list_predictions, epochs, property_identifier):
    """
    

    Parameters
    ----------
    list_predictions : List of lists
        List with several lists that contain property predictions.
        Each of these lists contains property predictions for a certain epoch
             
    property_identifier : String
        String that identifies the property 

    epochs : list
        Contains the epochs corresponding to the lists in list_predictions
    Returns
    -------
    None.

    """
    if property_identifier == 'kor':
        plot_title = 'Distribution of predicted pIC50 of generated molecules'
        plot_title = 'Distribution of predicted pIC50'
        label = 'Predicted pIC50'
    elif property_identifier == 'logP':
        plot_title = 'Distribution of predicted logP of generated molecules'
        label = 'Predicted logP'
    elif property_identifier == 'sascore':
        plot_title = 'Distribution of predicted SA score of generated molecules'
        label = 'Predicted SA score'
    elif property_identifier == 'mw':
        plot_title = 'Distribution of predicted MW of generated molecules'
        label = 'Predicted MW'
    elif property_identifier == 'qed':
        plot_title = 'Distribution of predicted QED of generated molecules'
        label = 'Predicted QED'
        
    legends = []
    sb.set_style("whitegrid")
    
    #palette = sb.husl_palette(6)
    #sb.set_style("whitegrid")
    #sb.set_palette("PuBu") #"YlGnBu  "YlOrRd" # "PuBu" "ocean" "YlOrBr"
    sb.set_palette("YlGnBu")
    
    #colors = ['#225ea8', '#41b6c4', '#a1dab4', '#ffffcc']
    
    #v1 = pd.Series(np.array(list_predictions[0]), name='Unbiased')
    v1 = pd.Series(np.array(list_predictions[0]), name=str(epochs[0]))
    legends.append('Unbiased')
    #legends.append(str(epochs[0]))
    ax = sb.kdeplot(v1, shade=True)
    #ax = sb.kdeplot(v1, shade=True, color = colors[0])
    for i in range(1,len(list_predictions)):
        v = pd.Series(np.array(list_predictions[i]), name='Epoch ' +str(epochs[i]))
        legends.append('Epoch ' +str(epochs[i]))
        #legends.append(str(epochs[i]))
        #sb.kdeplot(v, shade=True, color =colors[1])
        sb.kdeplot(v, shade=True)
        
    ax.set(xlabel=label,ylabel = 'Density', 
           title=plot_title)
    plt.legend(loc='upper right', labels=legends)
    plt.savefig('feedbackGAN_multiple_hist.png')
    plt.show()
    plt.close()
    

def drawMols(smiles, n_to_draw, info, property_identifier, predictor):
    """
    Function that draws chemical structure of the compounds generated by the optimized
    model.

    Parameters:
    -----------
    smiles: list with smiles strings
    n_to_draw: number of molecules to draw
    info: which molecules to draw ('random', 'max', 'min')
    property_identifier: main property of the experiment ('kor', 'QED')
    predictor: object that has .predict fucntion

    Returns
    -------
    This function returns a figure with the specified number of molecular 
    graphs indicating the pIC50 for KOR and the QED receptors.
    """
    DrawingOptions.atomLabelFontSize = 50
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 3
              
    
    # Elimina os compostos repetidos
    unique_smiles = list(np.unique(smiles))[1:]
    
    # prediction pIC50 KOR
    prediction_kor, og_idx = evaluate_property(predictor, unique_smiles, property_identifier)
    smiles = [unique_smiles[i] for i in og_idx]
    
    predictions_logP = evaluate_property(predictor, smiles, 'logP' )
    predictions_sascore = evaluate_property(predictor,smiles, 'sascore')
    predictions_QED = evaluate_property(predictor, smiles, 'qed')
    predictions_mw = evaluate_property(predictor, smiles, 'mw')
    
    
    mol_list = [Chem.MolFromSmiles(sm) for sm in smiles]
    #img = Draw.MolsToGridImage(mol_list)
    
    if info == 'random':
        # seleciona um numero previamente definido de moléculas (n_to_draw) para representar aleatoriamente
        ind = np.random.randint(0, len(smiles), n_to_draw)
        mols_to_draw = [mol_list[i] for i in ind]
        # coloca na legenda as propriedades desejadas para cada molécula
        legends = []
        for i in ind:
            legends.append('pIC50 for KOR: ' + str(round(prediction_kor[i],2))+ ' QED: ' + str(round(predictions_QED[i], 2)) + '\n logP: ' + str(round(predictions_logP[i], 2)) + ' SAscore: ' + str(round(predictions_sascore[i], 2)) + ' MW: ' + str(round(predictions_mw[i], 2)))
        
    else:
        
        
        prediction_kor_arr = np.array(prediction_kor)
        inds = prediction_kor_arr.argsort()
        mol_list_arr = np.array(mol_list)
        smiles_arr = np.array(smiles)
        
        sorted_mol_list = list(mol_list_arr[inds])
        sorted_smiles_list = list(smiles_arr[inds])
        l = len(sorted_mol_list)

        legends = []
        mols_to_draw = []
        if info == 'max':
            for i in range(n_to_draw):
                mols_to_draw.append(sorted_mol_list[l-i-1])
                prediction_kor, _ = evaluate_property(predictor, [sorted_smiles_list[l-i-1]], property_identifier)
                prediction_QED = evaluate_property(predictor, [sorted_smiles_list[l-i-1]], 'qed')
                prediction_logP = evaluate_property(predictor, [sorted_smiles_list[l-i-1]], 'logP')
                prediction_sascore = evaluate_property(predictor, [sorted_smiles_list[l-i-1]], 'sascore')
                prediction_mw = evaluate_property(predictor, [sorted_smiles_list[l-i-1]], 'mw')
                legends.append('pIC50 for KOR: ' + str(round(prediction_kor[0],2))+ ' QED: ' + str(round(prediction_QED[0], 2)) + '\n logP: ' + str(round(prediction_logP[0], 2)) + ' SAscore: ' + str(round(prediction_sascore[0], 2)) +' MW: ' + str(round(prediction_mw[0], 2)))
                
        elif info == 'min':
            for i in range(n_to_draw):
                mols_to_draw.append(sorted_mol_list[i])
                prediction_kor, _ = evaluate_property(predictor, [sorted_smiles_list[i]], property_identifier)
                prediction_QED = evaluate_property(predictor, [sorted_smiles_list[i]], 'qed')
                prediction_logP = evaluate_property(predictor, [sorted_smiles_list[i]], 'logP')
                prediction_sascore = evaluate_property(predictor, [sorted_smiles_list[i]], 'sascore')
                prediction_mw = evaluate_property(predictor, [sorted_smiles_list[i]], 'mw')
                legends.append('pIC50 for KOR: ' + str(round(prediction_kor[0],2))+ ' QED: ' + str(round(prediction_QED[0], 2)) + '\n logP: ' + str(round(prediction_logP[0], 2)) + ' SAscore: ' + str(round(prediction_sascore[0], 2))+ ' MW: '+ str(round(prediction_mw[0], 2)))
        
    
    
    
    img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=5, subImgSize=(300,300), legends=legends,returnPNG=False)
    img.save('mols_info_'+info+'_.png')
    img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=5, subImgSize=(300,300),returnPNG=False)
    img.save('mols_'+info+'.png')


def plot_QED_vs_SAscore(smiles):
    """
    Plots the QED values on the y-axis and the SAscore on the x-axis for the molecules in "SMILES"

    Parameters
    ----------
    smiles :list
        lsit with SMILES strings

    Returns
    -------
    None.

    """
    sb.set_palette("YlGnBu_r")
    sb.set_style("whitegrid")
    print(len(smiles))
    prediction_QED = evaluate_property('', smiles, 'qed')
    prediction_sascore = evaluate_property('', smiles, 'sascore')
    idx_rem = []
    for i in range(len(prediction_QED)):
        if prediction_QED[i] == 'NaN':
            idx_rem.append(i)
    
    for idx in (idx_rem):
        prediction_QED.pop(idx)
        prediction_sascore.pop(idx)
    #prediction_logP = evaluate_property('', smiles, 'logP')
    
    print(len(prediction_sascore))
    ax = sb.jointplot( prediction_sascore, prediction_QED, xlim = (0,10), ylim = (0,1), joint_kws={"s": 10})
    
    ax.set_axis_labels('SA score', 'QED', fontsize = 20)
    
    # plt.xlim(0,10.0)
    # plt.ylim(0,1.0)
    plt.savefig('feedbackGAN_QED_vs_SAscore.png')
    
def plot_QED_vs_SAscore_multi(smiles_1,smiles_2):
    """
    Plots the QED values on the y-axis and the SAscore on the x-axis for the molecules in "SMILES"

    Parameters
    ----------
    smiles :list
        lsit with SMILES strings

    Returns
    -------
    None.

    """
    
    # check if the smiles can be parsed into molecules by rdkit
    idx_rem  =[]
    for i,sm in enumerate(smiles_1):
        if Chem.MolFromSmiles(sm, sanitize=True) == None:
            idx_rem.append(i)
    
    for idx in (idx_rem):
        smiles_1.pop(idx)

    idx_rem  =[]
    for i,sm in enumerate(smiles_2):
        if Chem.MolFromSmiles(sm, sanitize=True) == None:
            idx_rem.append()
    
    for idx in (idx_rem):
        smiles_2.pop(idx)
    
    
    prediction_QED_1 = evaluate_property('', smiles_1, 'qed')
    prediction_sascore_1 = evaluate_property('', smiles_1, 'sascore')
    
    print(len(prediction_sascore_1))
    print(len(prediction_QED_1))
    idx_rem = []
    for i in range(len(prediction_QED_1)):
        if prediction_QED_1[i] == 'NaN':
            idx_rem.append(i)
    
    for idx in (idx_rem):
        prediction_QED_1.pop(idx)
        prediction_sascore_1.pop(idx)
    #prediction_logP = evaluate_property('', smiles, 'logP')

    assert len(prediction_QED_1) == len(prediction_sascore_1)
    
    prediction_QED_2 = evaluate_property('', smiles_2, 'qed')
    prediction_sascore_2 = evaluate_property('', smiles_2, 'sascore')
    idx_rem = []
    for i in range(len(prediction_QED_2)):
        if prediction_QED_2[i] == 'NaN':
            idx_rem.append(i)
    
    for idx in (idx_rem):
        prediction_QED_2.pop(idx)
        prediction_sascore_2.pop(idx)
    assert len(prediction_QED_2) == len(prediction_sascore_2)
    
    #label = np.squeeze(np.concatenate((np.ones((1, len(prediction_QED_1))), 2*np.ones((1, len(prediction_QED_2)))), axis = 1))
    label = ['Original Data' for i in range(len(prediction_QED_1))] + ['Generated Data' for i in range(len(prediction_QED_2))]
    pred_sascore = prediction_sascore_1 + prediction_sascore_2
    pred_QED = prediction_QED_1 + prediction_QED_2
    
    print(len(label))
    print(len(pred_sascore))
    print(len(pred_QED))
    d1 = {'Label':label, 'SA score': pred_sascore, 'QED':pred_QED}
    v1 = pd.DataFrame(d1)
    sb.set_palette("YlGnBu")
    ax = sb.relplot(data = v1, x = 'SA score', y= 'QED', hue = 'Label', palette = "YlGnBu_r", s=15)
    #sb.relplot(prediction_sascore_2, prediction_QED_2, xlim = (0,10), ylim = (0,1), joint_kws={"s": 10})
    ax.set(xlim = (0,10), ylim = (0,1))
    #ax.set_axis_labels('SA score', 'QED')
    #legends = ['Original data', 'Generated data']
    #plt.legend(loc='upper right', labels=legends)
    # plt.xlim(0,10.0)
    # plt.ylim(0,1.0)
    plt.savefig('feedbackGAN_QED_vs_SAscore_multi.png')
    
    
def plot_logP_vs_MW(smiles):
    """
    Plots the logP values on the y-axis and the MW on the x-axis for the molecules in "SMILES"

    Parameters
    ----------
    smiles :list
        lsit with SMILES strings

    Returns
    -------
    None.

    """
    sb.set_palette("YlGnBu_r")
    sb.set_style("whitegrid")
    print(len(smiles))
    prediction_logP = evaluate_property('', smiles, 'logP')
    prediction_MW = evaluate_property('', smiles, 'mw')
    # idx_rem = []
    # for i in range(len(prediction_QED)):
    #     if prediction_QED[i] == 'NaN':
    #         idx_rem.append(i)
    
    # for idx in (idx_rem):
    #     prediction_QED.pop(idx)
    #     prediction_sascore.pop(idx)
    #prediction_logP = evaluate_property('', smiles, 'logP')
    
    #print(len(prediction_sascore))
    ax = sb.jointplot( prediction_MW, prediction_logP,  joint_kws={"s": 10})
    
    ax.set_axis_labels('MW', 'LogP', fontsize = 20)
    
    # plt.xlim(0,10.0)
    # plt.ylim(0,1.0)
    plt.savefig('feedbackGAN_QED_vs_SAscore.png')

def plot_logP_vs_MW_multi(smiles_1,smiles_2):
    """
    Plots the QED values on the y-axis and the SAscore on the x-axis for the molecules in "SMILES"

    Parameters
    ----------
    smiles :list
        lsit with SMILES strings

    Returns
    -------
    None.

    """
    
    # check if the smiles can be parsed into molecules by rdkit
    idx_rem  =[]
    for i,sm in enumerate(smiles_1):
        if Chem.MolFromSmiles(sm, sanitize=True) == None:
            idx_rem.append(i)
    
    for idx in (idx_rem):
        smiles_1.pop(idx)

    idx_rem  =[]
    for i,sm in enumerate(smiles_2):
        if Chem.MolFromSmiles(sm, sanitize=True) == None:
            idx_rem.append()
    
    for idx in (idx_rem):
        smiles_2.pop(idx)
    
    
    prediction_logP_1 = evaluate_property('', smiles_1, 'logP')
    prediction_mw_1 = evaluate_property('', smiles_1, 'mw')
    
    print(len(prediction_mw_1))
    print(len(prediction_logP_1))
    idx_rem = []
    for i in range(len(prediction_logP_1)):
        if prediction_logP_1[i] == 'NaN':
            idx_rem.append(i)
    
    for idx in (idx_rem):
        prediction_logP_1.pop(idx)
        prediction_mw_1.pop(idx)
    #prediction_logP = evaluate_property('', smiles, 'logP')

    assert len(prediction_logP_1) == len(prediction_mw_1)
    
    prediction_logP_2 = evaluate_property('', smiles_2, 'logP')
    prediction_mw_2 = evaluate_property('', smiles_2, 'mw')
    idx_rem = []
    for i in range(len(prediction_logP_2)):
        if prediction_logP_2[i] == 'NaN':
            idx_rem.append(i)
    
    for idx in (idx_rem):
        prediction_logP_2.pop(idx)
        prediction_mw_2.pop(idx)
    assert len(prediction_logP_2) == len(prediction_mw_2)
    
    #label = np.squeeze(np.concatenate((np.ones((1, len(prediction_QED_1))), 2*np.ones((1, len(prediction_QED_2)))), axis = 1))
    label = ['Original Data' for i in range(len(prediction_logP_1))] + ['Generated Data' for i in range(len(prediction_logP_2))]
    pred_mw = prediction_mw_1 + prediction_mw_2
    pred_logP = prediction_logP_1 + prediction_logP_2
    
    d1 = {'Label':label, 'MW': pred_mw, 'LogP':pred_logP}
    v1 = pd.DataFrame(d1)
    sb.set_palette("YlGnBu")
    ax = sb.relplot(data = v1, x = 'MW', y= 'LogP', hue = 'Label', palette = "YlGnBu_r", s=15)
    #sb.relplot(prediction_sascore_2, prediction_QED_2, xlim = (0,10), ylim = (0,1), joint_kws={"s": 10})
    
    #ax.set_axis_labels('SA score', 'QED')
    #legends = ['Original data', 'Generated data']
    #plt.legend(loc='upper right', labels=legends)
    # plt.xlim(0,10.0)
    # plt.ylim(0,1.0)
    plt.savefig('feedbackGAN_logP_vs_MW_multi.png')

def eval_properties_to_csv(smiles, predictor):
    
    
    header = ['SMILES', 'KOR pIC50', 'QED', 'LogP', 'SAscore', 'MW', 'Mean TD', 'Median TD']
    
    prediction_kor, _  = evaluate_property(predictor, smiles, 'kor')
    prediction_QED = evaluate_property(predictor, smiles, 'qed')
    prediction_logP = evaluate_property(predictor, smiles, 'logP')
    prediction_sascore = evaluate_property(predictor, smiles, 'sascore')
    prediction_mw = evaluate_property(predictor,smiles, 'mw')
    
    with open('properties.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for i, sm in enumerate(smiles):
            # prediction_kor, _  = evaluate_property(predictor, [sm], 'kor')
            # prediction_QED = evaluate_property(predictor, [sm], 'qed')
            # prediction_logP = evaluate_property(predictor, [sm], 'logP')
            # prediction_sascore = evaluate_property(predictor, [sm], 'sascore')
            # prediction_mw = evaluate_property(predictor, [sm], 'mw')
            avg_tanimoto_distance, median_tanimoto_distance = eval_tanimoto_distance(sm, smiles)
            
            data = [sm, str(np.round(prediction_kor[i], decimals = 2)), str(np.round(prediction_QED[i],decimals = 2)), str(np.round(prediction_logP[i],decimals = 2)), str(np.round(prediction_sascore[i], decimals =2)), str(np.round(prediction_mw[i], decimals =2)), str(np.round(avg_tanimoto_distance,decimals = 2)), str(np.round(avg_tanimoto_distance,decimals = 2))]
            writer.writerow(data)
        
def eval_tanimoto_distance(sm, list_smiles):
    """
    Evaluates the average and median tanimoto distance between molecule A and every molecule

    Parameters
    ----------
    sm :  String
        SMILES string of molecule A
    list_smiles : list
        list with SMILES string of every molecule

    Returns
    -------
    mean_td : average tanimoto distance
    median_td : median of the tanimoto distance

    """
    
    fps_mol_A = AllChem.GetMorganFingerprint(MolFromSmiles(sm), 3)
    
    fps = []
    td = []
    for i, smile in enumerate(list_smiles):
        mol = MolFromSmiles(smile)
        fps_mol_i = AllChem.GetMorganFingerprint(mol, 3)
        ts = DataStructs.TanimotoSimilarity(fps_mol_A, fps_mol_i)
        td.append(1-ts)
    mean_td = np.mean(np.array(td))
    median_td = np.median(np.array(td))
    return mean_td, median_td