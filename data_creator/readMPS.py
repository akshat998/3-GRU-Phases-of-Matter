"""
    Please have a look at function readMPS
"""

import numpy as np


def read_indices(file_name='index.txt'):
    """(file_name) -> list
    
    Read file file_name line by line. Each line is stored as a sublist. 
    
    For a detailed explanation of 'index.txt', please view 
    'input_file_format_expln.ipynb. 
    
    @type file_name: str
    @rtype         : list
    """
    indices= []
    with open(file_name) as f:
        for line in f:
            int_list = [int(x) for x in line.split()]
            indices.append(int_list)
            
    return indices


def initialize_MPS_list(indices, num_tensors=16):
    """ (num_tensors, list) -> list
    
    Initialize a num_tensors length list with arrays of 0s. The shapes of arrays are
    determined by list indices.
    
    @type indices    : list
    @type num_tensors: int
    @rtype           : list
    """
    MPS = []
    
    for i in range(num_tensors):
        # End tensors have only 2 indices
        if i == 0 or i == num_tensors-1:
            MPS.append(np.zeros((indices[i][2],indices[i][3])))
            
        # All bulk tensors have 3 indices
        else:
            MPS.append(np.zeros((indices[i][2],indices[i][3],indices[i][4])))
        
    return MPS
    
    
def populate_tensors(MPS, MPS_populate, indices, num_tensors=16):
    """ (list, list, list) -> list
    
    Populate MPS with values contained in MPS_populate along indices stored in 
    indices. Value form MPS_populate is pasted in the right location of MPS.
    
    The end tensors have 2 indices, while everything in between (the bulk) have
    3 indices.
    
    @type MPS         : list
    @type MPS_populate: list
    @type indices     : list
    @type num_tensors : int
    @rtype            : list
    """
    counter=0
    
    
    # For first tensor (2 indices)
    for _ in range(indices[0][1]): # loop number of non-zero values of tensor
        MPS[0][ int(MPS_populate[counter][0])-1,int(MPS_populate[counter][1])-1]= MPS_populate[counter][2]
        counter+=1
    counter+=1 # skip hole in the file (Giacomo's format)

    # Populating bulk tensors (tensors 2...15)
    for i in range(1,num_tensors-1):
        for _ in range(indices[i][1]):  # loop number of non-zero values of tensor
            MPS[i][int(MPS_populate[counter][0])-1, int(MPS_populate[counter][1]) - 1, 
                   int(MPS_populate[counter][2])-1] = MPS_populate[counter][3]
            counter+=1
        counter+=1 # skip hole in the file (Giacomo's format)

    # For last tensor (2 indices)
    for _ in range(indices[num_tensors-1][1]): # loop number of non-zero values of tensor
        MPS[num_tensors-1][ int(MPS_populate[counter][0])-1,int(MPS_populate[counter][1])-1]= MPS_populate[counter][2]
        counter+=1
        
    return MPS


def permute_indexing(MPS, N=16):
    """ (list) -> list
    
    Indexing for N tensor is changed from a clockwise to an anticlockwise
    convention.
    
    @type MPS: list
    @rtype   : list
    """
    MPS[0] = np.transpose(MPS[0],(1,0))
    for i in range(1,N-1):
      MPS[i] = np.transpose(MPS[i],[2,1,0])
    return MPS      
    

def readMPS(MPSf, N=16,convert=True):
    """ (fileName, num_tensors, format_conversion) -> list
      
    Populate a list of N tensors with Matrix Product States of 'tensor.txt' based 
    on indices stored in 'index.txt'.
    
    tensor.txt & index.txt are stored in directory MPSf.
    For a detailed explanation of 'tensor.txt' & 'index.txt', please view 
    'input_file_format_expln.ipynb'
    
    @type MPSf     : str
    @type N        : int
    @type convert  : bool
    @rtype         : list
    """
    # Collect tensor indexes from 'index.txt'
    indices = read_indices(MPSf + '/index.txt')
            
    # Initialize right dimensions for subarrays of list (one for each tensor).
    MPS = initialize_MPS_list(indices) # dimensions are assigned based on indices
    
    # MPS_populate will be used to populate MPS
    MPS_populate = [np.array(list(map(float, line.split()))) for line in open(MPSf + '/tensor.txt')]

    # Populate MPS with values from MPS_populate
    MPS = populate_tensors(MPS, MPS_populate, indices)

    # Change tensor indexing from a clockwise to an anticlockwise convention.
    if convert == True:
        permute_indexing(MPS)

    return MPS

