import os
import numpy as np

# Pauli matrices
I = np.array([[1, 0],[0, 1]])

# First Pauli's matrix
pauli_X = np.array([[0, 1],[1, 0]])    

# Second Pauli's matrix
pauli_Y = np.array([[0, -1j],[1j, 0]]) 

# Third Pauli's matrix
pauli_Z = np.array([[1, 0],[0, -1]])   


def create_data_creation_interface():
    """
    Collect user entered parameters for creating data set via tensor network.
    
    The user enters:
        - POVM Type
        - Number of qubits
        - MPS type
        - An option to include local noise
        - Number of samples to be produced
        - Directory name for saving the data
        
    @rtype: (POVM_type, num_qubits, MPS_type, p, num_samples, path+folder_name)
    """
    while True:
        print()
        POVM_type = input("Please enter a POVM type. Choices include: 4Pauli/ Tetra/ Pauli/ Pauli_rebit/ Trine/ Psi2. \n").strip()
        if (POVM_type=='4Pauli' or POVM_type=='Tetra' or  POVM_type=='Pauli' or POVM_type=='Pauli_rebit' or POVM_type=='Trine' or POVM_type=='Psi2'):
            print()
            break
        print('Invalid Choice! Please choose among suggested chioices \n')
    
    while True:
        num_qubits = int(input("Please enter the number of qubits (must be greater than 2). \n").strip())  
        if (num_qubits >= 2):
            print()
            break
        print('Invalid Choice! Number of qubuts must be greater than 1. \n')
    
    while True:
        MPS_type = input("Please enter a MPS type. Choices include: \n 1. {} 2. {}".format(
                          'GHZ (Greenberger–Horne–Zeilinger state) \n',  
                          'Reading MPS from a user-provided file (in which case please provide the directory name)\n'))
        if (MPS_type == 'GHZ'):
            print()
            break

        elif (os.path.isdir(os.getcwd() + '/' + MPS_type) == True):            
            if ( os.path.isfile(MPS_type + '/tensor.txt' ) and os.path.isfile(MPS_type + '/index.txt' )):
                print()
                break
        print('Invalid Choice! Either use GHZ or a make sure that the custom directory exists with files index.txt & tensor.txt \n')
    
    while True:
        p = float(input("Want to include local noise with probability p? (0 = No noise). \n").strip())
        if (p >= 0 and p <= 1):
            print()
            break
        print('Invalid Input! Probability of including local noise must be between 0 and 1. \n')
    
    while True:
        num_samples = int(input("Please enter the number of samples in your data set. \n").strip())   
        if (num_samples > 0):
            print()
            break
        print('Invalid Input! Number of samples must me greater than 0. \n')
    
    while True:
        folder_name = input("Please enter name of directory for saving data (directory should not already exist inside 'DATA'). \n").strip()
        path = os.getcwd() + '/DATA/' # All folders are creted in directory 'DATA'
        
        if (os.path.isdir(path + folder_name) == False): # Make sure directory does not already exist
            os.mkdir(path + folder_name)
            print()
            break
        print('Directory already exists! please choose a unique name. \n')

    return POVM_type, num_qubits, MPS_type, p, num_samples, path+folder_name
    

def pXp(theta,phi):
    """ (float, float) -> array
    
    Apply a tranformation based on  angles phi and theta for obtaining Pauli POVM
    and PauliRebit POVM.
    
    @type theta: float
    @type phi  : float
    @rtype     : 2x2 array
    """
    return np.array([[ np.cos(theta/2.0)**2, np.cos(theta/2.0)*np.sin(theta/2.0)*np.exp(-1j*phi)],\
                     [ np.cos(theta/2.0)*np.sin(theta/2.0)*np.exp(1j*phi), np.sin(theta/2.0)**2 ]])

     
def mXm(theta,phi):
    """ (float, float) -> array
    
    Apply a tranformation based on  angles phi and theta for obtaining Pauli POVM
    and PauliRebit POVM.
    
    @type theta: float
    @type phi  : float
    @rtype     : 2x2 array
    """
    return np.array([[ np.sin(theta/2.0)**2, -np.cos(theta/2.0)*np.sin(theta/2.0)*np.exp(-1j*phi)],\
                     [-np.cos(theta/2.0)*np.sin(theta/2.0)*np.exp(1j*phi), np.cos(theta/2.0)**2 ]])


def construct_4Pauli_POVM(meas_outcomes=4):
    """ (int) -> numpy.ndarray
    
    Reconstruct 4 measurement 4-Pauli POVM(positive valued measurements)
    for a single qubit. Note that the overlap matrix of a 4-Pauli POVM is invertible. 
    
    Each line populates one matrix at a time
    For a deeper explanation, please refer to Page 16 of the paper.

    @type meas_outcomes: int
    @rtype             : ndarray
    """
    # Initialize 4 (2x2) matrices
    M = np.zeros((meas_outcomes, 2, 2), dtype=complex)
 
    # Copy in first 2x2 matrix
    M[0, :, :] = 1.0/3.0 * np.array([[1, 0],[0, 0]])
    
    # Copy in second 2x2 matrix
    M[1, :, :] = 1.0/6.0 * np.array([[1, 1],[1, 1]])
  
    # Copy in third 2x2 matrix
    M[2, :, :] = 1.0/6.0 * np.array([[1, -1j],[1j, 1]])
    
    # Copy in fourth 2x2 matrix
    M[3, :, :] = 1.0/3.0 * (np.array([[0, 0],[0, 1]]) + 0.5*np.array([[1, -1],[-1, 1]]) + 0.5*np.array([[1, 1j],[-1j, 1]]) )
    
    return M


def construct_Tetra_POVM(meas_outcomes=4):
    """ (int) -> numpy.ndarray
    
    Reconstruct a symmetric 4 measuremen Tetra POVM for a single qubit. 
    A linear combination of the pauli's matricies is added to the 2x2 identity,
    and  copied into the POVM array (M).

    For a deeper explanation, please refer to Page 17 of the Paper
    
    @type meas_outcomes: int
    @rtype             : ndarray
    """
    # Initialization of 4 (2x2) arrays
    M=np.zeros((meas_outcomes,2,2),dtype=complex)    
    
    # Copy in first 2x2 matrix
    M[0,:,:] = 1.0/4.0 * (I + pauli_Z)
    
    pauli_consts=np.array([2.0*np.sqrt(2.0)/3.0, 0.0, -1.0/3.0 ]) # constants used in linear combination of pauli's matrices
    M[1,:,:] = 1.0/4.0 * (I + (pauli_consts[0]*pauli_X) + (pauli_consts[1]*pauli_Y) + (pauli_consts[2]*pauli_Z) ) # Copy in second 2x2 matrix
    
    pauli_consts=np.array([-np.sqrt(2.0)/3.0 ,np.sqrt(2.0/3.0), -1.0/3.0 ]) # constants used in linear combination of pauli's matrices
    M[2,:,:] = 1.0/4.0 * (I + (pauli_consts[0]*pauli_X) + (pauli_consts[1]*pauli_Y) + (pauli_consts[2]*pauli_Z) ) # Copy in third 2x2 matrix
    
    pauli_consts=np.array([-np.sqrt(2.0)/3.0, -np.sqrt(2.0/3.0), -1.0/3.0 ]) # constants used in scaling of pauli's matrices
    M[3,:,:] = 1.0/4.0 * (I + (pauli_consts[0]*pauli_X) + (pauli_consts[1]*pauli_Y) + (pauli_consts[2]*pauli_Z) ) # Copy in forth 2x2 matrix
    
    return M


def construct_Pauli_POVM( Ps, theta, meas_outcomes=6):
    """ (ndarray, float, int) -> numpy.ndarray
    
    Reconstruct a 6 measurement Pauli POVM for a single qubit using probabilities
    Ps. Helper functions pXp and mXm are used to apply transformations based on 
    angles theta and phi.
   
    @type Ps           : 1-D numpy array
    @type theta:       : float
    @type meas_outcomes: int
    @rtype             : ndarray
    """
    # Initialization of 6 (2x2) arrays
    M = np.zeros((meas_outcomes,2,2),dtype=complex)
    
    # Copy in values for each matrix after scaling by probabilities
    M[0, :, :] = Ps[0] * pXp(theta, 0.0)
    M[1, :, :] = Ps[1] * mXm(theta, 0.0)
    M[2, :, :] = Ps[2] * pXp(theta, np.pi/2.0)
    M[3, :, :] = Ps[3] * mXm(theta, np.pi/2.0)
    M[4, :, :] = Ps[4] * pXp(0.0, 0.0)
    M[5, :, :] = Ps[5] * mXm(0, 0.0)

    return M
    

def construct_PauliRebit_POVM(Ps, theta, meas_outcomes=4):
    """ (ndarray, float, float, int) -> numpy.ndarray
    
    Reconstruct a 4 measurement Pauli POVM for a single qubit using probabilities
    Ps. 
    Helper functions pXp and mXm are used to apply transformations based on 
    angles theta and phi.
    
    @type Ps           : 1-D numpy array
    @type theta:       : float
    @type meas_outcomes: int
    @rtype             : ndarray
    """
    M = np.zeros((meas_outcomes, 2, 2),dtype=complex)
    
    # Copy in values for each matrix after scaling by probabilities
    M[0,:,:] = Ps[0]* pXp(theta,0.0)
    M[1,:,:] = Ps[1]* mXm(theta,0.0)
    M[2,:,:] = Ps[2]* pXp(0.0,0.0)
    M[3,:,:] = Ps[3]* mXm(0,0.0)
    
    return M


def construct_Trine_POVM(K=3):
    """ (int) -> numpy.ndarray
    
    Reconstruct a K-measurement outcome Trine POVM for a single qubit
    
    @type K : int
    @rtype  : ndarray
    """
    M = np.zeros((K,2,2),dtype=complex)
    phi0=0.0
    for k in range(K):
        phi =  phi0+ (k) * 2 * (np.pi / 3.0)
        M[k, :, :] = 0.5 * (I + (np.cos(phi) * pauli_Z) + np.sin(phi)* pauli_X)* (2 / 3.0)
    return M


def factory_POVMs(POVM_type):
    """ (str) -> (int, ndarray)
    
    Initialize positive valued measurements (POVMs) based on 
    POVM_type (e.g. Trine, Pauli, Tetra, etc.).
    
    The function returns the number of experimental outcomes and the construced
    POVM array.
    
    @type POVM_type: str
    @rtype         : int, ndarray

    """
    if POVM_type=='4Pauli':
        K = 4   # 4 measurement outcomes
        M = construct_4Pauli_POVM(meas_outcomes=K)              
 
    elif POVM_type=='Tetra':
        K=4     # 4 measurement outcomes
        M = construct_Tetra_POVM(meas_outcomes=K)
        
    elif POVM_type=='Pauli':
        K = 6   # 6 measurement outcomes
        Ps = np.array([1/3, 1/3, 1/3, 1/3, 1/3, 1/3]) 
        M = construct_Pauli_POVM(Ps, theta=np.pi/2.0)
        
    elif POVM_type=='Pauli_rebit':
        K = 4   # 4 measurement outcomes
        Ps = np.array([1./2., 1./2., 1./2., 1./2.])
        M = construct_PauliRebit_POVM(Ps, np.pi/2.0, meas_outcomes=K)

    elif POVM_type == 'Trine':
        K = 3    # 3 measurement outcomes
        M = construct_Trine_POVM()
   
    elif POVM_type == 'Psi2':    
        K = 2    # 2 measurement outcomes
        M = np.zeros((K,2,2),dtype=complex) 
        M[0,0,0] = 1
        M[1,1,1] = 1    

    return K, M