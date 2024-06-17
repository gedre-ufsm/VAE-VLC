Variational Autoencoder (VAE) for learning VLC channels with IQ complex dataset.

This code runs in Jupyter notebook v2024.4.0 with VSCode v1.89.1.

Dataset:
Need the files 'sent_data_tuple.npy', 'received_data_tuple_sync-phase.npy' that are the dataset to train the VAE.
The files 'sent_data_tuple.npy2', 'received_data_tuple_sync-phase.npy2' are dataset which I have changed the sample generation at acquisition (from 1MS to 100kS). So, you can use this second dataset to test the model.
The dataset folder has many other constellations to test (4psk, 16psk and 144psk).
The "full-disk" and "full-square" are dataset where I use no constellation, but random I/Q 16bits data normalized between -1 and 1. Where the objective is to cover a big partition of the transmission.

More details about the dataset will be published soon.

To use with GPU, you need to match the versions of Tensorflow, Python, cuDNN and CUDA. https://www.tensorflow.org/install/source#gpu

## This version bellow is outdate.
Collab Version: [[https://colab.research.google.com/drive/18TvTYv3Nxv6eK0Hv8iOdDAB71fL7wcDh?usp=sharing](https://colab.research.google.com/drive/1IFp5BourVsgvqeTb-NFPGmbRx3s7eYeY?usp=sharing)](https://colab.research.google.com/drive/18TvTYv3Nxv6eK0Hv8iOdDAB71fL7wcDh?usp=sharing)
##

The point is to replicate the VLC channel behavior. The EVM and the constellation are the metrics.
