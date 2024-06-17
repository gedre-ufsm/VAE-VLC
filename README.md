Variational Autoencoder (VAE) for learning VLC channels with IQ complex dataset.

This code runs in Jupyter notebook v2024.4.0 with VSCode v1.89.1.

Need the files 'sent_data_tuple.npy', 'received_data_tuple_sync-phase.npy' that are the dataset to train the VAE.
The files 'sent_data_tuple.npy2', 'received_data_tuple_sync-phase.npy2' are dataset within I have change the samples generation at aquisition (from 1MS to 100kS). So, you can use this second dataset to test the model. 

To use with GPU, you need to match the versions of Tensorflow, Python, cuDNN and CUDA. https://www.tensorflow.org/install/source#gpu

Collab Version: [[https://colab.research.google.com/drive/18TvTYv3Nxv6eK0Hv8iOdDAB71fL7wcDh?usp=sharing](https://colab.research.google.com/drive/1IFp5BourVsgvqeTb-NFPGmbRx3s7eYeY?usp=sharing)](https://colab.research.google.com/drive/18TvTYv3Nxv6eK0Hv8iOdDAB71fL7wcDh?usp=sharing)

The point is to replicate the VLC channel behavior. The EVM and the constellation are the metrics.
