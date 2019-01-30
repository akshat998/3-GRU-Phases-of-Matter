### This part will inform you about the data generation process.
---

### Prerequisites:

* Please ensure you have cloned the repository and are in the directory:

         `3-GRU-Phases-of-Matter`.

* Further, cd into data_creator:

         `cd data_creator/`

* Run the command:

         `python noisy_generation.py`

  This will initiate a light-weight GUI, asking you about parameters for the data generation process.


### Parameters:

The user will be prompted to enter the following parameters upon running noisy_generation.py:

* POVM types: 4Pauli, Tetra, Pauli, Pauli_rebit, Trine, Psi2.
* Number of qubits
* MPS type  : GHZ (or importing a uder-defined MPS file)
* Adding local noise with a certain probability
* The size of the data set (number of samples)
* Directory name for saving all the data

Please note that all data is stored in the directory:
    `data_creator`

### Importing A Custom MPS File:
Check out [these instructions.](docs/custom_mps_HowTo.ipynb)

### What Tensor Netwrk is responsible for the data generation?
Check out this [Back-End explanation](docs/data_gen_expln.ipynb)
