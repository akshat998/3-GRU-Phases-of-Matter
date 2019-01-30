import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd.variable import Variable
from utils import * 
import os
from sampling import *
from data_creator.utils import factory_POVMs
import numpy as np


batch_size = 5


class Model(nn.Module):

    def __init__(self):
        """ (None) -> None
        
        Initialize the layers of our model. 
        We make use of 3 stacked GRU cells followed by a linear layer.
        """
        
        super(Model, self).__init__()
        n_features = N * K   
        n_out = N * K
        
        # THREE STACKED GRU CELLS
        
        # Stack-1
        self.rnn_1 = nn.GRU(n_features, layer_size, num_layers=1, dropout=0.30, batch_first=True)
        self.h_1 = self.initialize_hidden(layer_size)
        
        # Stack-2
        self.rnn_2 = nn.GRU(layer_size, layer_size, num_layers=1, dropout=0.30, batch_first=True)
        self.h_2 = self.initialize_hidden(layer_size)
        
        # Stack-3
        self.rnn_3 = nn.GRU(layer_size, layer_size, num_layers=1, dropout=0.30, batch_first=True)
        self.h_3 = self.initialize_hidden(layer_size)
        
        # Linear layer
        self.linear = nn.Sequential(
                nn.Linear(layer_size, n_out),
                nn.Dropout(0.30)
        )
        

    def forward(self, x, N, K):
        """ (ndarray) -> ndarray
        
        Return the output of a foward pass of our Network. A Softmax layer is 
        applied based on the number of qubits N and measurement outcomes K.
        
        The input(x) goes through:
            - 3 stacked GRU Cells
            - A linear layer
            - A softmax layer based on the number of measurement outcomes
            
        @type N : int
        @type K : int
        @type x : ndarray
        @rtype  : ndarray
        """        
        num_samples = x.shape[0]
        x = x.unsqueeze(0)
        
        # Flatten parameters of all stack members
        self.rnn_1.flatten_parameters()
        self.rnn_2.flatten_parameters()
        self.rnn_3.flatten_parameters()
        
        sigmoid = nn.Sigmoid() # Pass each output through a non-linearity

        # Stack component - 1
        out, self.h_1 = self.rnn_1(x, self.h_1)
        out = sigmoid(out)
        
        # Stack component - 2
        out, self.h_2 = self.rnn_2(out, self.h_2)
        out = sigmoid(out)
        
        # Stack component - 3
        out, self.h_3 = self.rnn_3(out, self.h_3)
        out = sigmoid(out)
        
        # Pass through a Linear layer
        out = self.linear(out) 
        
        # Followed by a Softmax: Based on the number of measurement outcomes
        out = out.reshape((num_samples, N, K))
        softmax = nn.Softmax(2) 

        # Reshape output to original dimensions
        out = softmax(out).reshape((num_samples, N * K)) 
        return out.reshape((num_samples, N, K))


    def initialize_hidden(self, rnn3_layer_size):
        """
        Initialize the hidden state of the RNN at time step 0.
        
        A tensor is returned with dimension:
            (Number of GRU Cells, 1, Layer size of the GRU Cell)
        
        @type rnn3_layer_size: int
        @rtype               : PyTorch Variable
        """
        return Variable(torch.randn(1, 1, rnn3_layer_size), requires_grad=True) 



def train_RNN(optimizer, batch, model, N, epoch):
    """ (torch.optim.Adam, ndarray, RNN_Torch.Kodel)
    
    Perform backpropogation on model using the Adam optimizer based on the ideal 
    behaviour of data batch. 
    
    Number od qubits N ensures the right dimension of output from training.
    ross-Entropy loss is used as the loss function
    
    @type N        : Number of qubits
    @type optimizer: torch.optim.Adam
    @type batch    : ndarray
    @type model    : RNN_Torch.Model
    """
    K, M = factory_POVMs('Tetra')
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Sample RNN
    prediction = model(torch.zeros((batch_size, N*K)), N, K)
    prediction = prediction.reshape((batch_size, N*K)) 
    
    # Cross-Entropy loss
    error = - ( (batch * torch.log(prediction) )) 
    error = (error.sum(1)).mean()
    
    # Perform backpropogation on network
    error.backward(retain_graph=True)
    nn.utils.clip_grad_norm(model.parameters(), 0.5) # Apply Gradient Clipping
    optimizer.step() # Update parameters
    
    return error


def run_model(num_epochs, N):
    """ (int) -> Folder containing saved model

    Train our model using backpropagation. Training time of the model depends 
    on the number of epochs.
    
    @type num_epochs: int
    @rtype          : None
    """
    model = Model()
    
    if torch.cuda.is_available(): # Train on GPU, if available
        model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
    for epoch in range(num_epochs): 
            
        for n_batch, batch in enumerate(data_loader):
            
            batch = Variable(batch.reshape(batch.shape[0], N*K)) 
            
            if torch.cuda.is_available(): 
                batch = batch.cuda()
                
            error = train_RNN(optimizer, batch, model, N, epoch)
            print('Batch[{}/{}] Error:{} '.format(n_batch, num_batches, error.data.cpu().numpy()))
            
        save_model(model, save_model_name, str(epoch))

        # TODO: TESTING
#        _, M = factory_POVMs('Tetra')
        
        # TODO
#        predictions = sample_model(save_model_name + '_' + str(epoch), 100000, N, K)
        
        # Report classical fidelity
#        histogram = calc_fid(predictions, 100000, 'A', N, K)
#        histogram = np.asarray(histogram).reshape((K ** N, 1))
#        print(histogram)
    
    # Report quantum fidelity        
#        if N <= 6:
#            den_recons, den_orig = calc_quant_fid(histogram, M, K, N)     
#            print('den_recons = ', den_recons)
#            print('den_orig = ', den_orig)
    
#            eigenvalues, _ = np.linalg.eigh(den_recons)   
#            print('Reconstr. eigenvalies: ', eigenvalues)
#            eigenvalues, _ = np.linalg.eigh(den_orig)   
#            print('Original. eigenvalies: ', eigenvalues)
        
#            print('Positivity of density matrix ', all(i >= 0 for i in eigenvalues))
         
#        else:
#            Fidelity(np.asanyarray(predictions), M, N) # Monte Carlo approximation (scalable)


    
    # Save trained model
    print("Epochs of Model ", save_model_name, " has been saved in directory 'saved_models'")
    
    
if __name__ == '__main__':
    
    # User interface for training parameter
    data_folder, num_epochs, layer_size, save_model_name = create_training_interface()
        
    # Importing number of ubits from pickled file (created upon initiating data creation)
    pickle_in = open('data_creator/DATA/' + data_folder + '/vars.pickle', 'rb')
    N = pickle.load(pickle_in)[1]
    pickle_in.close()
    
    data = read_data_set(data_folder)
    data = torch.tensor(data, dtype=torch.float)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    num_batches = len(data_loader)
    
    K = int((data.shape[1]) / N) # Number of measurement outcomes
    
    run_model(num_epochs, N) 


