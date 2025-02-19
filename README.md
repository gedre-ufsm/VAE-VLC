Variational Autoencoder (VAe) for learning VLC channels with IQ complex dataset.

This code runs in Jupyter notebook v2024.4.0 with VSCode v1.89.1.

The file "VAe_simple" uses the dataset folder to train the model, where the files 'sent_data_tuple.npy' and 'received_data_tuple_sync-phase.npy' are stored.

The "dataset" folder contains data for four types of modulation (4-QAM, 16-QAM, 64-QAM, and 256-QAM) and the "full_square". We train the model using the full_square dataset and then use this trained model to compare with different modulation schemes.

The file "importos" is used to print the constellations from the .npy data files in the same folder for visual inspection.

The "Models" folder contains all pre-trained models used for comparison in the qualification thesis.

The file "VAE_Xqam_KL_HISTOGRAM" uses the dataset folder and applies all trained models to determine the best-performing one.

GPU Compatibility
To use this code with a GPU, you must ensure that TensorFlow, Python, cuDNN, and CUDA versions are compatible. More details:. https://www.tensorflow.org/install/source#gpu

## This version bellow is outdate.
Collab Version: [[https://colab.research.google.com/drive/18TvTYv3Nxv6eK0Hv8iOdDAB71fL7wcDh?usp=sharing](https://colab.research.google.com/drive/1IFp5BourVsgvqeTb-NFPGmbRx3s7eYeY?usp=sharing)](https://colab.research.google.com/drive/18TvTYv3Nxv6eK0Hv8iOdDAB71fL7wcDh?usp=sharing)
##

