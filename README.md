# Vae-GAN
Vae-GAN to VLC with IQ complex dataset

This code run in VSC. 
Need the files 'sent_adjusted.npy' and 'received_adjusted.npy'.
The data are in this format:
Shape of IQ_x_complex: (500000,)
First 10 values of IQ_x_complex: [0.66937494-0.8225552j   0.6490789 -0.62534857j  0.42154524-0.6299593j 0.512511  -0.929957j   -0.09680751+0.14961769j -0.4932171 +0.96642506j -0.6960994 +0.41515794j  0.5434779 +0.67704326j  0.55949014+0.0441226j -0.7632868 +0.70712423j]

To use the GPUyou need match the versions of Tensorflow, Python, cuDNN and CUDA. https://www.tensorflow.org/install/source#gpu

Collab Version: https://colab.research.google.com/drive/18TvTYv3Nxv6eK0Hv8iOdDAB71fL7wcDh?usp=sharing

