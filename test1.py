#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python


# In[1]:


'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from csv import reader

from keras.layers import Lambda, Input, Dense
# keras lambda layer
from keras.models import Model
# from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# from keras.layers import Input, Dense
# from keras.models import Model
import csv
from os import listdir
from os.path import isfile, join
from scipy.misc import imread
# import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random as rd


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon


# In[2]:

def classcrossentropy(inputs,outputs):
    print(type(inputs))
    if(runmode == "single"):
        return binary_crossentropy(inputs,outputs)

    imlabel = 0
    i1 = 0
    for im1 in flowerlist:
        if(im1 == inputs):
            imlabel = labels[i1]
            break
        i1 = i1+1
    print(imlabel)
#     for key in dict1:    
    loss = 0
    i1 = 0
    for image1 in flowerlist:
        if(labels[i1] == imlabel):
            loss = loss + binary_crossentropy(image1,outputs)
        i1 += 1
    return loss


# In[2]:


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# In[3]:


# In[3]:


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# In[6]:


# In[4]:
# folder = "jpg"
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# In[9]:


dict1 = {}
mypath = "/home/anurag/Desktop/projects/VAEproj/102flowers/jpg"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
filename = 'FileName.csv'

labels = load_csv(filename)
labels = labels[0]    
print(onlyfiles)
# i1 = 1
# seltol = 1300
#seltol = 10
# selected1 = 800
# for selected1 in [200]:
# samplpositive = 600
#samplpositive = 10
featurestotal = []
i1 = 0
for image in onlyfiles:
    if(image[-3:] != "jpg"):
        continue
    # i1 = i1 + 1
    # print()
    im1 = mypath+"/"+ image
    # print(im1)
    im = imread(im1)
    res = cv2.resize(im, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
    #res = cv2.resize(im, dsize=(500,500))
    # res = im
    k1 = np.reshape(res,-1)
    k1 = list(k1)
#     dict1[k1] = labels[i1] 
    # print(k1.shape)
        # print(i1)
    print(i1)
    featurestotal.append(k1)
    i1 = i1+ 1
    # if(i1 == seltol + 1):
        # break


flowerlist = []
rd.shuffle(featurestotal)
flowerlist.extend(featurestotal)

x_train = featurestotal[:7000]
x_test = featurestotal[7000:]
x_train = np.array(x_train)
x_test = np.array(x_test)

# MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


image_size = x_train[0].shape[0]
print(image_size)
original_dim = image_size * image_size
# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (original_dim, )
print("input_shape")
print(input_shape)
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    # dict1 = {}
    kllosswt = 1;
    parser = argparse.ArgumentParser()

    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

# here1
    # VAE loss = mse_loss or xent_loss + kl_loss
    
    runmode = input("single or class encoder ")



    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = classcrossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss = kl_loss*-0.5*kllosswt
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    # plot_model(vae,
               # to_file='vae_mlp.png',
               # show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_mnist.h5')


    # plot_results(models,
                 # data,
                 # batch_size=batch_size,
                 # model_name="vae_mlp")


# In[ ]:


# In[ ]:




