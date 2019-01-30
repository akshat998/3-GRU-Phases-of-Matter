import torch
from torch.autograd.variable import Variable
import os
import numpy as np
from utils import create_sampling_interface, read_data_set, Kron, buildT, get_hist_keys
from RNN_Torch import *
import itertools as it              
from scipy.linalg import sqrtm
from data_creator.utils import factory_POVMs
import warnings  
from data_creator.ncon import ncon
warnings.filterwarnings('ignore')
import pickle


def sample_model(rnn_file_name, num_samples, N, K):
    """ (str, int, int, int) -> tensor
    
    Sample saved rnn of file rnn_file_name (num_samples are generated).
    The rnn is expected to have been trained on an N qubit system with M 
    measurement outcomes.
    
    @type rnn_file_name: str
    @type num_sample   : int
    @type N            : int
    @type K            : int
    @rtype             : list
    """
    rnn_dir = './saved_models/' + rnn_file_name # Path of saved model
    model = torch.load(rnn_dir, map_location='cpu') # Conversion from gpu->cpu instace required
    
    prediction = model(torch.zeros((num_samples, N*K)), N, K)
    prediction = torch.argmax(prediction.reshape((num_samples, N, K)), 2)
    return prediction
    

def calc_fid(predictions, num_samples, data_file, N, K):
    """ (ndarry, int, str) -> list
    
    Calculate classifical fidelity, comparing samples generated from the RNN
    and the ones generated from the tensor network.
    
    The final Classical fidelity is printed & histogram of the RNN samples 
    is returned. 
    
    @type predictions: ndarray
    @type num_samples: int
    @type datafile   : str
    @rtype           : list
    """
    actual_samples = read_data_set(data_file, 'data.txt') # TODO
    
    sampling_prob = get_hist_keys(N, K) # TODO
    
    # Calculating histogram of actual data
    for item in actual_samples:
        if str(item) in sampling_prob:
            sampling_prob[str(item)][0] += 1
    
    # Calculating histogram of the RNN
    for item in predictions:
        s = str(item.numpy()).replace(',', '')
        if s in sampling_prob:
            sampling_prob[s][1] += 1

    
    # Convert into probabilities
    for key, value in sampling_prob.items():
        sampling_prob[key][0] = sampling_prob[key][0] / actual_samples.shape[0]
        sampling_prob[key][1] = sampling_prob[key][1] / num_samples 
        
    # Calculate similarity
    classical_fid = 0
    for key, value in sampling_prob.items():
        classical_fid += np.sqrt(sampling_prob[key][0] * sampling_prob[key][1])
    
    print('Classical Fidelity: ', classical_fid)
    
    keys = sorted(list(sampling_prob.keys()))
    histogram = []
    for item in keys:
        histogram.append(sampling_prob[item][1])
        
    return histogram



def calc_quant_fid(histogram, M, K, N):
    """
    Calculate quantum fidelity using the overlap matrix.
    
    Note:
        This method is not scalable for large qubits. A Monte-Carlo approximation
        is used for larger that 6 qubit systems
    """
    p= 0.0 # TODO: Serialuze original file to get this stuff more easily 

    alpha=1./np.sqrt(2.)
    beta=1./np.sqrt(2.)
    norm=np.sqrt(np.abs(alpha)**2+np.abs(beta)**2)
    alpha=alpha/norm
    beta=beta/norm
    Z = np.array([[1, 0],[0, -1]])
    
    # magnetization operator
    mz=np.zeros((2**N,2**N))
    for k in range(N):
        op=Kron([Z],[k],N)
        mz=mz+op
    
    mm = (np.diag(mz)+N) / 2.  
    
    lam = np.zeros((N+1,1))
    
    for i in range(N+1):
        lam[i] = abs(alpha)**2*((1.0-p/2.0)**(N-i))*((p/2.0 )**i)+abs(beta)**2*((1.0-p/2.0)**(i))*((p/2)**(N-i))
    
    indi = list(range(2**N+2))
    indj = list(range(2**N+2))
    
    elements = np.zeros(2**N+2)
    
    for i in range(2**N):
        elements[i]=lam[mm[i].astype(int)]
    
    indi[2**N] = 0
    indj[2**N] = 2**N-1
    elements[2**N] = alpha*np.conj(beta)*(1-p)**N
    
    indi[2**N+1] = 2**N-1
    indj[2**N+1] = 0
    elements[2**N+1] = np.conj(alpha)*(beta)*(1-p)**N
    
    rho = np.zeros((2**N, 2**N))    
    rho[indi,indj] = elements
        
    a = (np.array(list(it.product(list(range(K)), repeat = N)))) # basis set ``bosonic'' occupation
    P = np.zeros(a.shape[0])
    
    ListMa=[]
    for i in range(a.shape[0]):
        Ma = M[a[i,0]]
        #print float(i)/a.shape[0] 
        for j in range(1,N):
            Ma = np.kron(Ma,M[a[i,j]])
    
        ListMa+=[Ma]
        P[i] = np.trace(np.matmul(Ma,rho))
    
    P = P/np.sum(P)

    tMaMa = np.zeros((4,4))
    for i in range(4): 
        for j in range(4):
            tMaMa[i,j] = np.trace(np.matmul(M[i], M[j]))
    
    ## building T matrix explicitly
    T = np.zeros((K**N,K**N))
    for i in range(a.shape[0]):
        for j in range(i,a.shape[0]):
                T[i,j] = buildT(tMaMa,i,j,N,a)
                T[j,i] = T[i,j]   
    
    eps=1e-12
    rho=rho+eps*np.eye(2**N,2**N )
    
    Pr = histogram
    Qa = np.matmul(np.linalg.inv(T), Pr) 
    
    rho_rec = np.zeros((2**N,2**N))
    for i in range(K**N):
        rho_rec = rho_rec + ListMa[i]*Qa[i]
    
    Fidelity = np.trace(sqrtm(np.matmul(sqrtm(rho), np.matmul(rho_rec,sqrtm(rho)))))
    print ('Quantum Fidelity ', abs(Fidelity))
    
    return rho_rec, rho  
    


def Fidelity(S, M, N):
    """
    A Monte-Carlo approximation of Quantum Fidelity. This method is scalable
    for larger qubit systems.
    """
    t = ncon((M,M),([-1,1,2],[ -2,2,1]))
    it = np.linalg.inv(t)
    
    
    # TODO: For GHZ
    cc = np.zeros((2,2)) # corner
    cc[0,0] = 2**(-1.0/(2*N))
    cc[1,1] = 2**(-1.0/(2*N))
    cb = np.zeros((2,2,2)) # bulk
    cb[0,0,0] = 2**(-1.0/(2*N))
    cb[1,1,1] = 2**(-1.0/(2*N))

   
    MPS = []
    MPS.append(cc)
    for i in range(N-2):
        MPS.append(cb)
    MPS.append(cc) 
    
    Fidelity = 0.0
    F2 = 0.0
    Ns = S.shape[0]
    
    for i in range(Ns):

        # contracting the entire TN for each sample S[i,:]  
        eT = ncon((it[:,S[i,0]],M,MPS[0],MPS[0]),([3],[3,2,1],[1,-1],[2,-2]))

        for j in range(1,N-1):
            eT = ncon((eT,it[:,S[i,j]],M,MPS[j],MPS[j]),([2,4],[1],[1,5,3],[2,3,-1],[4,5,-2]))      

        j = N-1
        eT = ncon((eT,it[:,S[i,j]],M,MPS[j],MPS[j]),([2,5],[1],[1,4,3],[3,2],[4,5]))
        Fidelity = Fidelity + eT
        F2 = F2 + eT**2
        Fest=Fidelity/float(i+1)
        F2est=F2/float(i+1)
        Error = np.sqrt( np.abs( F2est-Fest**2 )/float(i+1))
        
        
    F2 = F2/float(Ns)
    Fidelity = np.abs(Fidelity/float(Ns))
    Error = np.sqrt( np.abs( F2-Fidelity**2 )/float(Ns))
    print('Fid: ', np.real(Fidelity), ' Error: ',  Error)
    print()


if __name__ == '__main__':  
    print('Sampling model...')  
    # Remove N, POVM 
    model_path, num_samples, original_data_file = create_sampling_interface()
    
    pickle_in = open('data_creator/DATA/' + original_data_file + '/vars.pickle', 'rb')
    data = pickle.load(pickle_in) # Stored pickled data
    POVM = data[0]
    N = data[1]
    pickle_in.close()
 
    
    K, M = factory_POVMs(POVM)
    
    predictions = sample_model(model_path, num_samples, N, K)
    
    # Report classical fidelity
    histogram = calc_fid(predictions, num_samples, original_data_file, N, K)

    histogram = np.asarray(histogram).reshape((K ** N, 1))
    
    # Report quantum fidelity
    if N <= 6:
        rho_rec, rho = calc_quant_fid(histogram, M, K, N)         # Direct Method (not scalable)
        eigenvalues, _ = np.linalg.eigh(rho)    
        print('Original. eigenvalies: ', eigenvalues)
        eigenvalues, _ = np.linalg.eigh(rho_rec)   
        print('Reconstc. eigenvalies: ', eigenvalues)
    else:
        Fidelity(np.asanyarray(predictions), M, N) # Monte Carlo approximation (scalable)



