# Introduction

This repository uses a Recurrent Neural Network (with 3-stacked GRU cells) for reconstructs quantum states/

This implementation is based on the findings of the following [paper](https://arxiv.org/abs/1810.10584) .

# Installation Requirements

Before running the code, please ensure you have the following:

- [Python 3.0](https://www.python.org/download/releases/3.0/)
- [Pytorch v0.4.1](https://pytorch.org/)

# Getting Started

For a quick start, please ensure the following.

- Clone the repository: In an appropriate directory run the following command on your Terminal:

  `git clone https://github.com/akshat998/3-GRU-Phases-of-Matter.git`

- Make sure you `cd` into the right directory.

  `cd 3-GRU-Phases-of-Matter/`

- [Create your data](https://github.com/akshat998/3-GRU-Phases-of-Matter/tree/master/data_creator)

  By following the above instructions, the data will be saved in a folder of directory:

  `/data_creator/DATA`

  Please ensure you remember the name of your data directory.

- Train your model:

  The Recurrent Neural Network has been created using PyTorch. To start training,
  please run the following command on your Terminal:

  ` python3 RNN_Torch.py`

  You will be asked to enter some parameters prior to training.

  Error is printed for each batch & after each batch, the model is saved in the directory (with format `EnteredName_batchNumber`):

   `saved_models`

- Evaluate your model:

  After training your model, you can test the quality of your model (i.e. calculate Classical & Quantum Fidelity). To do so, please run:

  ` python3 sampling.py`

  Upon running, the user will be asked to provide:
  * Name of the saved model
  * Number of samples to be generated from the saved model
  * Name of the original data file that was used to train the saved mode

  Please note:
  For smaller systems, Quantum Fidelity is calculated using the overlap matrix.
  And for larger systems, Quantum Fidelity is calculated using a Monte-Carlo Approximation

# Questions, problems?

Make a github issue 😄. Please be as clear and descriptive as possible. Please feel free to reach
out in person: (akshat[DOT]nigam[AT]mail[DOT]utoronto[DOT]ca)

# Acknowledgment
* A special thanks to the author of the [paper](https://arxiv.org/abs/1810.10584): [Dr. Juan Carrasquilla](https://vectorinstitute.ai/team/juan-felipe-carrasquilla/)

* Also, the creator of ncon.py: **__________**
