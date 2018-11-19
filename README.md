# Image-Domain-Adaptation

Based on Haeusser's 'Learning by Association' and 'Associative Domain Adaptation', a convolutional net for transfering 
knowledge between two image domains
Original two papers:
https://arxiv.org/abs/1706.00909
https://arxiv.org/abs/1708.00938

The included MNIST-SVHN.py file uses labelled SVHN and unlabelled MNIST images for training/testing.
The MNIST files are downloaded via Keras in the code; the SVHN .mat files need to be added to the directory manually.
To run, make sure all requirements are installed (as detailed in requirements.txt, plus CUDNN), then run:
  python MNIST-SVHN.py
or, more generally, 
  python <filename>
followed by desired commandline arguments. 
