# Voice-Activity-Detection
We train a neural network to detect activity of human speech in an audio frame. 

The audio dataset can be found here at http://www.openslr.org/resources.php 

A soft VAD value is computed and used from the speech signal and some random mixtures with chunks of noise signals for each frame of speech data.

Take Speech frames and extract Log spectrogram or MFCCs to get the features ready to be used as inputs for the neural network.

Build 
(a) Feedforward Model
(b) RNNs with LSTMs

The model has been trained with MSE Cost function, ReLU output layer and RMSProp optimization for approximately 20-50 epochs and a dropout and decayed learning rate alongside 10% validation error. 

TODO: 
1. Better Hyperparameter optimization. 
2. Explore more complex topologies. 
3. Use better strategies to come up with generalizations
