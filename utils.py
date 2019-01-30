import numpy as np
import csv
import os
import torch
from data_creator.utils import factory_POVMs
import itertools


def read_data_set(folder_name, file='train.txt'):
    """ (str) -> ndarray
    
    A functions for reading data in 'folder_name/train.txt'. This data is supplied to the 
    model for training.
    
    @type folder_name: str
    @rtype           : ndarray
    """
    with open('data_creator/DATA/' + folder_name + '/' +file, newline='\n') as inputfile:
        results = list(csv.reader(inputfile))
        
    gather = []    
    for item in results:
        for item_1 in item:
            a = []
            for char in item_1:
                if char != ' ':
                    a.append(int(char))
            gather.append(a)
                
    return np.asarray(gather)


def save_model(model, file_name, epoch): # TODO: epoch
    """ (RNN_Torch.Model, str) -> None
    
    Save trained RNN onto folder 'saved_models' as 'file_name'
    
    @type model: RNN_Torch.Model
    @rtype     : Output file directory
    """
    torch.save(model, './saved_models/{}'.format(file_name + '_' + epoch))    
    
    
def create_training_interface():
    """
    Collect user entered parameters for training RNN. 
    
    The user enters:
        - Folder name containg the training data
        - Number of epochs for training the mode
        - Layer size for the 3 stacked GRU cells
        - A name for saving the trained model
        
    @rtype: (data_folder, num_epochs, layer_size, save_model_name)
    """
    # DATA Folder Name
    while True:
        data_folder = input("Please enter folder name which contains generated data. The Folder must be contained in directory: data_creator/DATA. \n").strip()
        path = os.getcwd() + '/data_creator/DATA/' # All models are saved in directory 'saved_models'

        if (os.path.isdir(path + data_folder) == True): # Make sure folder exists
            print('')
            break
        print("Folder does not exist. Please check directory 'data_creator/DATA' and try again  \n")
    

    # Get the number of epochs a user would like to train the model
    while True:
        print()
        num_epochs = input("Please enter the number of epochs you would like to train the model \n").strip()
        try:
            num_epochs = int(num_epochs)
            break
        except:
            print("Invalid choice. Please make sure that you enter a number.")
            
    # Get the layer size of the stacked GRU cells        
    while True:
        print()
        layer_size = input("Please enter the layer size of the GRU cells \n").strip()
        try:
            layer_size = int(layer_size)
            break
        except:
            print("Invalid choice. Please make sure that you enter a number.")
    
    # save model name
    while True:
        print()
        save_model_name = input("Please enter a name under which the model will be saved (All models are saved inside directory saved_models). \n").strip()
        path = os.getcwd() + '/saved_models/' # All models are saved in directory 'saved_models'
        
        if (os.path.isfile(path + save_model_name) == False): # Make sure file does not already exist
            break
        print('File already exists! please choose a unique name. \n')

    return data_folder, num_epochs, layer_size, save_model_name


def create_sampling_interface():
    """
    Collect user entered parameters for sampling already trainined RNN. 
    
    The user enters:
        - Saved model file name
        - Number of samples to be generated from model
        - Original data folder
        
    @rtype: (model_path, num_samples, N, POVM_type, data_folder)
    """
    # Get saved RNN file name
    while True:
        model_path = input("Please enter name of saved rnn. The Folder must be contained in directory: 'saved_models'. \n").strip()    
        if (os.path.isfile('./saved_models/' + model_path) == True): # Make sure folder exists
            print('')
            break
        print("Saved model does not exist. Please check directory 'saved_models' and try again  \n")
    
    # Get number of samples  that will be generated from the RNN
    while True:
        print()
        num_samples = input("Please enter the number of samples you would like generate from the rnn \n").strip()
        try:
            num_samples = int(num_samples)
            break
        except:
            print("Invalid choice. Please make sure that you enter a number.")             
            
    # Get original data folder
    while True:
        data_folder = input("Please enter folder name which contains generated data. The Folder must be contained in directory: data_creator/DATA. \n").strip()
        path = os.getcwd() + '/data_creator/DATA/' # All models are saved in directory 'saved_models'

        if (os.path.isdir(path + data_folder) == True): # Make sure folder exists
            print('')
            break
        print("Folder does not exist. Please check directory 'data_creator/DATA' and try again  \n")
        
    return model_path, num_samples, data_folder


def get_hist_keys(N, K):
    """ (N, K) -> dict
    
    Return a dictionary whose keys are possible states for an N qubit system 
    with K measurement outcomes.
    
    For N=5, K=4 -> len(output) = 4**5 = 1024
    
    @type N: int
    @type K: int
    @rtype : dict
    """
    outcomes = [i for i in range(K)] # Single qubit states
    
    qb_states = list(itertools.product(outcomes, repeat=N)) # Combinations for multipe qubit states
    gather = []
    for item in qb_states:
        s = ''.join(str(element)+' ' for element in item) 
        gather.append('[' + s.strip() + ']')
        
    possible_states = {}
    for item in gather:
        possible_states[item] = [0, 0]
        
    
    return possible_states
        


def Kron( operators, position, N ):
    I = np.eye(2)
    count = 0
    if position[0]==0:
       out = operators[0]
       count += 1 
    else:
        out = I


    for i in range(1,N): 
        if i in position:
           out = np.kron(out,operators[count])
           count += 1
        elif i not in position:
           out = np.kron(out,I)

    return out


def buildT(tMaMa,i,j,N,a):
    out=1.0
    for x in range(N):
        out *= tMaMa[a[i,x],a[j,x]]
    return out 