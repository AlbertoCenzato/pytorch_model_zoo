# pytorch_model_zoo
A repository for my pytorch models. It mainly contains recurrent models:
* LSTMCell stack: as PyTorch LSTM but it can be called with the same semantics of LSTMCell, i.e. it processes one sequence element at a time instead of an entire sequence.
* LSTM stack: as PyTorch LSTM but allows for hidden layers of different size.
* ConvLSTM: an implementation of [Shi et al. ConvLSTM](http://arxiv.org/abs/1506.04214), instead of fully connected layers for the LSTM gates it uses convolutional layers.
* LSTM autoencoder: a sequence-to-sequence model similar to the model proposed in [Sutskever et al. _Sequence to Sequence Learning with Neural Networks_](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf). The model consists of one encoder LSTM an one (or more) decoder LSTMs. The encoder reads the whole sequence and compresses it in an intermediate representation which is then used by the decoder(s) to produce a new sequence.  
* ConvLSTM autoencoder: same idea as the LSTM autoencoder but using ConvLSTMs instead of LSTMs.  