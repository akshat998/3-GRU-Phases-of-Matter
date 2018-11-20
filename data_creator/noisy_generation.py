import numpy as np
from ncon import ncon
from  readMPS import readMPS
import os

# Pauli matrices
I = np.array([[1, 0],[0, 1]])

# First Pauli's matrix
pauli_X = np.array([[0, 1],[1, 0]])    

# Second Pauli's matrix
pauli_Y = np.array([[0, -1j],[1j, 0]]) 

# Third Pauli's matrix
pauli_Z = np.array([[1, 0],[0, -1]])   


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

def construct_MPS(MPS_type, num_qubits):
    """ (str, int) -> ndarray
    
    Construct matrix product states based on the number of qubits. For GHZ, tensors 
    are coppied in. For all other cases, MPS values are read in according to format
    in tensor.txt
    
    @type MPS_type  : str
    @type num_qubits: int
    @rtype          : ndarray
    """
    MPS = []
    
    if MPS_type=='GHZ': 
        # Copy tensors used to construct GHZ as an MPS. The procedure below should work for any other MPS 
        GHZ_value =  2 **(-1.0/(2 * num_qubits))
        
        # Corner tensors (for begining and end)
        corner_tensor = np.zeros((2,2)) # corner
        corner_tensor[0,0] = GHZ_value
        corner_tensor[1,1] = GHZ_value
        
        # Bulk tensors
        bulk_tensors = np.zeros((2,2,2)) 
        bulk_tensors[0,0,0] = 2 **(-1.0/(2 * num_qubits))
        bulk_tensors[1,1,1] = 2 **(-1.0/(2 * num_qubits))
    
        # Add in corner and bulk matrices
        MPS.append(corner_tensor)
        for i in range(num_qubits - 2):
            MPS.append(bulk_tensors)
        MPS.append(corner_tensor) 
        
    else: 
        MPS = readMPS(MPSf=MPS_type, N=num_qubits,convert=True)      

    return MPS


def write_generated_data(state, num_exp_outcomes, num_qubits, dir_name):
    """ (ndarray, int, str, str) -> None
    
    Write evaluated state of tensor network with num_exp_outcomes experimental outcomes
    onto files:
        1. data_file   : Contains raw data of state
        2. one_hot_file: Contains one hot encoding of state. Used as a training 
                         data set
                         
    @type state           : ndarray
    @type num_exp_outcomes: int
    @type data_file       : str
    @type one_hot_file    : strs
    @rtype                : None
    """
    f_1 = open(dir_name + '/data.txt', 'a+')
    f_2 = open(dir_name + '/train.txt', 'a+')
    
    # Generate a one hot encoding of data in state
    one_hot = np.squeeze(np.reshape(np.eye(num_exp_outcomes)[state],[1,num_qubits * num_exp_outcomes]).astype(np.uint8)).tolist()
    
    # Write raw data 
    for item in state:
        f_1.write("%s " % item)
    f_1.write('\n')
    f_1.flush()

    # Write data training(one hot encoded) data
    for item in one_hot:
        f_2.write("%s " % item)  
    f_2.write('\n') 
    f_2.flush() 
 
    
def construct_noisyMPO_from_MPS(p):
    """ (float) -> ndarray
    
    Apply local noise p converting matrix product state(MPS) to noisy 
    matrix product operator (MPO)
    
    @type p : float
    @rtype  : ndarray
    """
    noisyMPO = np.zeros((2,2,4,4))

    E00 = np.zeros((4,4))
    E00[0,0] = 1

    E10 = np.zeros((4,4))
    E10[1,0] = 1

    E20 = np.zeros((4,4))
    E20[2,0] = 1

    E30 = np.zeros((4,4))
    E30[3,0] = 1

    noisyMPO = noisyMPO + np.sqrt(1.0-p) * ncon((I,       E00), ([-1,-2],[-3,-4]))
    noisyMPO = noisyMPO + np.sqrt(p/3.0) * ncon((pauli_X, E10),([-1,-2],[-3,-4]))
    noisyMPO = noisyMPO + np.sqrt(p/3.0) * ncon((pauli_Y, E20),([-1,-2],[-3,-4]))
    noisyMPO = noisyMPO + np.sqrt(p/3.0) * ncon((pauli_Z, E30),([-1,-2],[-3,-4]))

    return noisyMPO


class PaMPS():

    
    def __init__(self, POVM_type='Trine', num_qubits=4, MPS_type='GHZ', p=0.0):
        """  (str, int, str, float) -> Output data onto files
        
        Perform repeated tensor operations which lead to the production of our data.
        
        Tensor operations are implemented for:
               - POVM Types : 4Pauli, Tetra, Pauli, Pauli_rebit, Trine, Psi2
               - MPS Types  : GHZ / reading MPS from user-provided files
               - num_qubits : Greater than to 2 qubits
               - Adding local depolarizing noise with probability p
            
        @type POVM_type : string (4Pauli/ Tetra/ Pauli/ Pauli_rebit/ Trine/ Psi2)
        @type num_qubits: int    (greater than 2)
        @type MPS_type  : String (GHZ/ read from custom file )
        @type p         : float  (between 0 and 1)
        @rtype          : None
        """
        E0 = np.zeros((4))
        E0[0] = 1
        
        noisyMPO = construct_noisyMPO_from_MPS(p) # constructing noisy MPO from the MPS 
        
        self.N = num_qubits
        self.num_exp_outcomes, self.POVM = factory_POVMs(POVM_type)
        self.MPS = construct_MPS(MPS_type, num_qubits)
        self.locMixer = ncon((noisyMPO, E0, np.conj(noisyMPO), E0), ([-1,-2,1,3],[3],[-4,-3,1,2],[2]))
        self.LocxM = ncon((self.POVM, self.locMixer), ([-3,1,2], [2,-2,-1,1]))
        self.l_P = self.construct_contract_prob()

 
    def construct_contract_prob(self):
        """ (None) -> ndarray
        
        Constuct initial probability array(l_P), which will be used in repeated contraction
        to genrate more samples. 
        
        @rtype: ndarray
        """
        Tr = np.ones((self.num_exp_outcomes)) 
        
        l_P = [None] * self.N
        l_P[self.N-1] = ncon((self.POVM, self.MPS[self.N-1], self.MPS[self.N-1], self.locMixer),\
                                  ([-3,1,2],        [3,-1 ],        [4,-2],            [2,4,3,1]))
        
        for i in range(self.N-2,0,-1):
            l_P[i] = ncon((self.POVM  , self.MPS[i], self.MPS[i], self.locMixer, l_P[i+1], Tr),\
                               ([-3,4,5],    [-1,6,2],    [-2,7,3],     [5,7,6,4],     [2,3,1 ],     [1]))
        
        l_P[0] = ncon((self.POVM,   self.MPS[0], self.MPS[0], self.locMixer, l_P[1], Tr),\
                           ([-1,4,5 ],      [6,2],      [7,3],       [5,7,6,4],     [2,3,1 ],  [1]))
        return l_P
    
    
    def gen_samples(self, Ns, dir_name):
        
        """ (int) -> Data saved onto text files
        
        Loop through number of samples Ns, performing multiple contractions
        that eventually get written onto 2 text files (one with raw data & other
        with a on-hot version). 
         
        @type Ns: int
        @rtype  : None
        """
        state = np.zeros((self.N),dtype=np.uint8)
        
        print ('Generating Samples...')
        
        for  _ in range(Ns):
            Pi = np.real(self.l_P[0]) 
            Pnum = Pi            
            i=1
            state[0] = (np.argmax(np.random.multinomial(n=1, pvals=Pi,size=1)))   
            Pden = Pnum[state[0]]
            PP = ncon((self.POVM[state[0]], self.locMixer, self.MPS[0], self.MPS[0] ),\
                          ([2,1],            [1,4,3,2],       [3,-1],      [4,-2]))    

            for i in range(1,self.N-1):  
                Pnum = np.real(ncon((PP,self.l_P[i]), ([1,2],[1,2,-1])))
                Pi   = Pnum/Pden
                state[i] =  np.argmax(np.random.multinomial(n=1, pvals=Pi,size=1)) 
                Pden = Pnum[state[i]]
                PP =  ncon((PP,  self.LocxM[:,:,state[i]], self.MPS[i], self.MPS[i] ),\
                           ([1,2],    [3,4],               [1,3,-1],     [2,4,-2]))
              
            i = self.N-1
            Pnum = np.real(ncon((PP, self.l_P[self.N-1]),([1,2], [1,2,-1])))
            Pi   =   Pnum / Pden
            state[self.N-1] = np.argmax(np.random.multinomial(n=1, pvals=Pi,size=1)) 
            
            # Write raw data onto 'data.txt' and one hot encoding onto 'train.txt'
            write_generated_data(state, self.num_exp_outcomes, self.N, dir_name)
            
         

if __name__ == "__main__":    
    # Gather inputs and perform error checking...
    
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
                          'Reading MPS from a user-provided file (in which case please provide the directory name)\n'
                         ))
        if (MPS_type == 'GHZ'):
            print()
            break
        # TODO: CHECK FOR EXISTENCE OF index.txt & tensor.txt
        if (os.path.isdir(os.getcwd() + '/' + MPS_type) == True):
            print()
            break
        
        print('Invalid Choice! Either use GHZ or a make sure that the custom directory exists \n')
    
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

    # Generate samples
    sampler = PaMPS(POVM_type, num_qubits, MPS_type, p)
    samples = sampler.gen_samples(num_samples, path+folder_name)
    


